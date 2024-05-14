from osocrNG.names import default_ocr_variable_names
class default_hydra_variable_names(default_ocr_variable_names):

    @ classmethod
    def prefix(cls,prefix,term):
        return prefix+term;


