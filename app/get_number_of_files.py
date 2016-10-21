import os

def get_number_of_files(training=True):
    if training == True:
        folder = "set_train"
    else:
        folder = "set_test"
    DIR_IN = os.path.join(
        DATA_DIRECTORY,
        folder
        )
    nof = len([name for name in os.listdir(DIR_IN) if os.path.isfile(os.path.join(DIR_IN, name))])
    return nof
