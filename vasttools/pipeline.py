import pandas as pd
import os
import warnings
import glob


class Pipeline(object):
    """Class to interface with VAST Pipeline results"""
    def __init__(self, project_dir):
        super(Pipeline, self).__init__()

        self.project_dir = os.path.abspath(project_dir)

    def load_run(self, runname):
        """
        Load a pipeline run.
        """

        run_dir = os.path.join(
            self.project_dir,
            runname
        )

        if not os.path.isdir(run_dir):
            raise ValueError(
                "Run '%s' does not exist!",
                runname
            )
            return

        associations = pd.read_parquet(
            os.path.join(
                run_dir,
                'associations.parquet'
            )
        )

        images = pd.read_parquet(
            os.path.join(
                run_dir,
                'images.parquet'
            )
        )

        skyregions = pd.read_parquet(
            os.path.join(
                run_dir,
                'skyregions.parquet'
            )
        )

        sources = pd.read_parquet(
            os.path.join(
                run_dir,
                'sources.parquet'
            )
        )

        m_files = images['measurements_path'].tolist()

        m_files += sorted(glob.glob(os.path.join(
            run_dir,
            "forced_measurements*.parquet"
        )))

        measurements = pd.read_parquet(
            m_files[0]
        )

        for i in m_files[1:]:
            measurements = measurements.append(
                pd.read_parquet(i)
            )

        measurements = measurements.merge(
            associations, left_on='id', right_on='meas_id',
            how='left'
        ).drop([
            'meas_id',
            'd2d_x',
            'dr_x'
        ], axis=1).rename(
            columns={
                'source_id': 'source',
                'd2d_y': 'd2d',
                'dr_y': 'dr'
            }
        ).reset_index(drop=True)

        piperun = PipeRun(
            name=runname,
            associations=associations,
            images=images,
            skyregions=skyregions,
            sources=sources,
            measurements=measurements
        )

        return piperun


class PipeRun(object):
    """An individual pipeline run"""
    def __init__(
        self, name=None, associations=None, images=None,
        skyregions=None, sources=None, measurements=None
    ):
        super(PipeRun, self).__init__()
        self.name = name
        self.associations = associations
        self.images = images
        self.skyregions = skyregions
        self.sources = sources
        self.measurements = measurements
