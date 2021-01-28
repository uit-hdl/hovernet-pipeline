import os
from tensorpack.tfutils.export import ModelExporter
from config import Config
from misc.utils import rm_n_mkdir

class Exporter(Config):

    def run(self): 
        exporter = ModelExporter(get_pred_config())
        rm_n_mkdir(self.model_export_dir)
        exporter.export_compact(filename='{}/compact.pb'.format(self.model_export_dir))
        exporter.export_serving(os.path.join(self.model_export_dir, 'serving'), signature_name='serving_default')
        print(f'Saved model to {self.model_export_dir}.')


if __name__ == '__main__':
    exporter = Exporter()
    exporter.run()