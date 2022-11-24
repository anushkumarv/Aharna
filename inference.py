from config import config
from feature_extractors_inference.dev_phase_inference import QryInf

inf_obj = QryInf(config)
inf_obj.infer_data()