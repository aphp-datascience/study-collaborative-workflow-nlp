CSES = ["cse180032", "cse200055", "cse200093"]
PATIENT_TYPES = ["inpatient", "outpatient"]
LABEL_TYPES = ["ml", "rule_based"]
MODELS = ["eds", "base"]

CKPTS = dict(
    base=dict(
        # Overall="/data/scratch/tpetitjean/ML/checkpoints_test/tune-head:simple-t_lr:5e-05-h_lr:0.0005-valid_f1:0.9525.ckpt",
        Overall="/data/scratch/tpetitjean/ML/checkpoints_test/training_camembert_base-head:simple-t_lr:2e-05-h_lr:0.001-valid_f1:0.9444.ckpt",
        cse180032="/data/scratch/tpetitjean/ML/checkpoints_test/base/training_camembert_base-cse:cse180032-valid_f1:0.9417.ckpt",
        cse200055="/data/scratch/tpetitjean/ML/checkpoints_test/base/training_camembert_base-cse:cse200055-valid_f1:0.9583.ckpt",
        cse200093="/data/scratch/tpetitjean/ML/checkpoints_test/base/training_camembert_base-cse:cse200093-valid_f1:0.9378.ckpt",
    ),
    eds=dict(
        Overall="/data/scratch/tpetitjean/ML/checkpoints_test/train_final_camembert_eds-head:simple-t_lr:4e-05-h_lr:0.001-valid_f1:0.9450.ckpt",
        cse180032="/data/scratch/tpetitjean/ML/checkpoints_test/eds/training_camembert_eds-cse:cse180032-valid_f1:0.9484.ckpt",
        cse200055="/data/scratch/tpetitjean/ML/checkpoints_test/eds/training_camembert_eds-cse:cse200055-valid_f1:0.9748.ckpt",
        cse200093="/data/scratch/tpetitjean/ML/checkpoints_test/eds/training_camembert_eds-cse:cse200093-valid_f1:0.9583.ckpt",
    ),
)

RARE_COMORBS = ["Hémiplégie", "SIDA"]

TRIGRAM_MAPPING = {
    "014": "APR",
    "028": "ABC",
    "095": "AVC",
    "005": "BJN",
    "009": "BRK",
    "010": "BCT",
    "011": "BCH",
    "033": "BRT",
    "016": "BRC",
    "042": "CFX",
    "019": "CRC",
    "021": "CCH",
    "022": "CCL",
    "029": "ERX",
    "036": "GCL",
    "075": "EGP",
    "038": "HND",
    "026": "HMN",
    "099": "HAD",
    "041": "HTD",
    "032": "JVR",
    "044": "JFR",
    "047": "LRB",
    "049": "LRG",
    "053": "LMR",
    "061": "NCK",
    "096": "PBR",
    "066": "PSL",
    "068": "RPC",
    "069": "RMB",
    "070": "RDB",
    "072": "RTH",
    "073": "SAT",
    "079": "SPR",
    "076": "SLS",
    "084": "SSL",
    "087": "TNN",
    "088": "TRS",
    "090": "VGR",
    "064": "VPD",
    "INC": "INCONNU",
}
