from pathlib import Path

project_path = Path(__file__).resolve().parents[1]

class Paths:
    """
    Facilitates access to useful paths, to project base directory and other folders
    """

    def __init__(self):
        """ Define the paths """

        self.project_dir = str(project_path)
        self.codebase_dir = f'{project_path}/codebase/'
        self.data_dir = f'{self.codebase_dir}data/'
        self.notebook_dir = f'{self.codebase_dir}notebooks/'
        self.crosswalk_dir = f'{project_path}/supplementary_online_data/ONET_ESCO_crosswalk/'
        self.figure_dir = f'{self.codebase_dir}/reports/figures/'
        self.models_dir = f'{self.codebase_dir}/models/'
        self.config_dir = f'{self.codebase_dir}/configs/'

        self.dict_of_all_paths = {
            'project_dir': self.project_dir,
            'codebase_dir': self.codebase_dir,
            'data_dir': self.data_dir,
            'notebook_dir': self.notebook_dir,
            'crosswalk_dir': self.crosswalk_dir,
            'figure_dir': self.figure_dir,
            'models': self.models_dir,
            'configs': self.config_dir,
        }

    def list_all(self):
        """ Display all available useful paths """

        for p in list(self.dict_of_all_paths.keys()):
            print(p)


class Viz:
    """
    Colour palletes [DEPRECATED]
    """

    def __init__(self):
        self.color_pal = [
            '#DCDCDC', '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
            '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
            '#008080', '#e6beff', '#9a6324', '#808080', '#800000',
            '#aaffc3', '#000000', '#ffd8b1', '#808000', '#000075']
