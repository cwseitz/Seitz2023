from tiler import Tiler
from correct import Basic
from detect import Detector
#from nucleus_model import NucleusModel
#from cell_model import CellBodyModel
from segment import CellSegmenter
from pathlib import Path

class Pipeline:
    def __init__(self,config,prefix):
        self.config = config
        self.datapath = config['datapath']
        self.analpath = config['analpath']
        self.cmodelpath = config['cmodelpath']
        self.nmodelpath = config['nmodelpath']
        self.cell_filters = config['cell_filters']
        self.nucleus_filters = config['nucleus_filters']
        self.p0 = config['p0']
        self.prefix = prefix
        Path(self.analpath+self.prefix).mkdir(parents=True, exist_ok=True)
    def execute(self):
        self.tile()
        self.basic_correct()
        self.apply_nucleus_model()
        self.apply_cell_model()
        self.detect_spots()
    def tile(self):
        tiler = Tiler(self.datapath,self.analpath,self.prefix)
        tiler.tile()
    def basic_correct(self):
        basic = Basic(self.analpath,self.prefix)
        basic.correct()
    def apply_nucleus_model(self):
        nmodel = NucleusModel(self.nmodelpath,self.analpath,self.prefix,self.nucleus_filters)
        nmodel.apply()
    def apply_cell_model(self):
        cmodel = CellBodyModel(self.cmodelpath,self.analpath,self.prefix,self.cell_filters)
        cmodel.apply()
    def detect_spots(self):
        detector = Detector(self.datapath,self.analpath,self.prefix)
        detector.detect()
    def segment_cells(self):
        cellsegment = CellSegmenter(self.datapath,self.analpath,self.prefix,self.cell_filters,self.p0)
        cellsegment.segment()


