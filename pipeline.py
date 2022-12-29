from tiler import Tiler
from correct import Basic
from detect import Detector
from nucleus import NucleusModel
from cellbody import CellBodyModel
from pathlib import Path

class Pipeline:
    def __init__(self,config,prefix):
        self.config = config
        self.datapath = config['datapath']
        self.analpath = config['analpath']
        self.cmodelpath = config['cmodelpath']
        self.nmodelpath = config['nmodelpath']
        self.prefix = prefix
        Path(self.analpath+self.prefix).mkdir(parents=True, exist_ok=True)
    def execute(self):
        self.tile()
        self.basic_correct()
        self.segment_nuclei()
        self.segment_cells()
        self.detect_spots()
    def tile(self):
        tiler = Tiler(self.datapath,self.analpath,self.prefix)
        tiler.tile()
    def basic_correct(self):
        basic = Basic(self.analpath,self.prefix)
        basic.correct()
    def segment_nuclei(self):
        nmodel = NucleusModel(self.nmodelpath,self.analpath,self.prefix)
        nmodel.segment()
    def segment_cells(self):
        cmodel = CellBodyModel(self.cmodelpath,self.analpath,self.prefix)
        cmodel.segment()
    def detect_spots(self):
        detector = Detector(self.datapath,self.prefix)
        detector.detect()


