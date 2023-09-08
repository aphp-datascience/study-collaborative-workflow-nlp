from ecci.icd10 import ICD10Getter


def update_icd10():
    icd10_getter = ICD10Getter()
    icd10_getter.get()


if __name__ == "__main__":
    update_icd10()
