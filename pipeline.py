from tiler import Tiler
from correct import Basic
from detect import Detector
from nucleus import NucleusModel
from cellbody import CellBodyModel

class Pipeline:
    def __init__(self,config,prefix):
        self.config = config
        self.datapath = config['datapath']
        self.analpath = config['analpath']
        self.cmodelpath = cconfig['modelpath']
        self.nmodelpath = config['nmodelpath']
        self.prefix = prefix
    def execute(self):
        self.tile()
        self.basic_correct()
        self.segment_nuclei()
        self.segment_cells()
        self.detect_spots()
        self.summarize()
    def tile(self):
        tiler = Tiler()
        tiler.Tile()
    def basic_correct(self):
        basic = Basic()
        basic.correct()
    def segment_nuclei(self):
        nmodel = NucleusModel()
    def segment_cells(self):
        cmodel = CellBodyModel()
    def detect_spots(self):
        detector = Detector()
        detector.detect()


