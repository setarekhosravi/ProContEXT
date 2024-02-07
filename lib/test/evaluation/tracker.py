import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv
from datetime import datetime

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np

now = datetime.now()
now = now.strftime("%Y-%m-%d_%H_%M_%S")

_tracker_disp_colors = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0),
                        4: (255, 255, 255), 5: (0, 0, 0), 6: (0, 255, 128),
                        7: (123, 123, 123), 8: (255, 128, 0), 9: (128, 0, 255)}


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False, gt_path: str = None, update_rate: int = None):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()
        frame_number = 0

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        def get_ground_truth(path):   
            file = open(path, "r")
            gTruth = []
            for item in file:
                item = item.strip()
                item = item.strip(";")
                item = eval(item)
                gTruth.append(item)

            return gTruth

        def groundTruth():
            path = gt_path
            gTruth = get_ground_truth(path)
            for j in range(0,len(gTruth)):
                if len(gTruth[j])!=0:
                    gTruth[j] = gTruth[j][1:5]

            return gTruth
        
        if update_rate is None:
            raise ValueError("Unknown update_rate, you should enter value")
        else:
            gt_check = True

        gTruth = groundTruth()
        output_boxes = []
        non_gt = False

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            # output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)
                if not gt_path:
                    x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                else:
                    if len(gTruth[frame_number])!=0:
                        non_gt = False
                        x, y, w, h = gTruth[frame_number]
                    else:
                        non_gt = True
                        init_state = [1, 1, 10, 10]
                        initial_state=tracker.initialize(frame, _build_init_info(init_state))

                        while len(gTruth[frame_number])==0:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame_number +=1
                            save = []
                            output_boxes.append(save)
                        x, y, w, h = gTruth[frame_number]

                        non_gt = False
                init_state = [x, y, w, h]
                initial_state=tracker.initialize(frame, _build_init_info(init_state))
                save_state = [0.0, x, y, x+w, y+h, 1]
                output_boxes.append(save_state)
                break

        first_time = True
        time_list = []
        while True:
            t1 = time.time()
            ret, frame = cap.read()

            if frame is None:
                break

            frame_number += 1
            frame_disp = frame.copy()

            if frame_number%update_rate == 0:
                gt_check = True
            else:
                gt_check = False

            if first_time:
                out = tracker.track(frame, initial_state)
                non_gt = False
                first_time = False

            if gt_check:
                if len(gTruth[frame_number])==0:
                    non_gt = True
                
                else:
                    non_gt = False
                    x, y, w, h = gTruth[frame_number]
                    gt_state = [x, y, w, h]
                    out = tracker.initialize(frame,_build_init_info(gt_state))


            # Draw box
            if not non_gt:
                out = tracker.track(frame)         
                state =[int(s) for s in out['target_bbox']]
                # save = [0]
                # save.extend([out['target_bbox'][0], out['target_bbox'][1], out['target_bbox'][2] + out['target_bbox'][0], out['target_bbox'][3] + out['target_bbox'][1], 1])    
                # print(save)
                # output_boxes.append(save)
                if gt_check:
                    cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                                _tracker_disp_colors[2], 5)
                else:
                    cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                                _tracker_disp_colors[1], 5)


            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
            t2 = time.time()
            time_list.append(t2-t1)

            bboxes = out['target_bbox']
            conf = out["score"]
            cls = 0.0
            
            if non_gt==False:
                output_boxes.append([cls,bboxes[0],bboxes[1],bboxes[0]+bboxes[2],bboxes[1]+bboxes[3],conf])

            if non_gt:
                save=[]
                output_boxes.append(save)


        # When everything done, release the capture
        avg_time = np.sum(time_list)/len(time_list)
        print("Avg inference time: ", avg_time, " FPS: ", 1/avg_time)
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            if not os.path.exists(base_results_path):
                os.makedirs(base_results_path)
            bbox_file = '{}.txt'.format(base_results_path+'/'+video_name+now)
            with open (bbox_file, 'w+') as tx:
                for i in output_boxes:
                    if len(i)==0:
                        tx.write(str(i)+'\n')
                    else:
                        tx.write('['+str(i)+']'+'\n')

    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")
