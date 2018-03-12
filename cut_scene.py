import scenedetect

class PySceneDetectArgs(object):
	def __init__(self, input, type='content'):
		self.input = input
		self.detection_method = type
		self.threshold = None
		self.min_percent = 95
		self.min_scene_len = 15
		self.block_size = 8
		self.fade_bias = 0
		self.downscale_factor = 1
		self.frame_skip = 2
		self.save_images = False
		self.start_time = None
		self.end_time = None
		self.duration = None
		self.quiet_mode = True
		self.stats_file = None

video_path = './data/test2.mp4'
scene_detectors = scenedetect.detectors.get_available()

smgr_content   = scenedetect.manager.SceneManager(PySceneDetectArgs(input=video_path, type='content'),   scene_detectors)
#smgr_threshold = scenedetect.manager.SceneManager(PySceneDetectArgs(input=video_path, type='threshold'), scene_detectors)

video_fps, frames_read, frames_processed = scenedetect.detect_scenes_file(path=video_path, scene_manager=smgr_content)
#scenedetect.detect_scenes_file(path=video_path, scene_manager=smgr_threshold)
scene_list = smgr_content.scene_list
# create new list with scene boundaries in milliseconds instead of frame #.
scene_list_msec = [(1000.0 * x) / float(video_fps) for x in scene_list]
# create new list with scene boundaries in timecode strings ("HH:MM:SS.nnn").
scene_list_tc = [scenedetect.timecodes.get_string(x) for x in scene_list_msec]
print(scene_list_tc)

