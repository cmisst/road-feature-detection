import pandas as pd
import numpy as np

class PredictLable:
    df = None

    def __init__(self, *args, **kwargs):
        self.df = pd.read_csv(kwargs['csv_src'])
        self.df.set_index('BEPT', inplace=True)

    def load_column(self, **kwargs):
        col_names = ['BEPT', kwargs['col_name']]
        if 'csv_src' in kwargs:
            df = pd.read_csv(kwargs['csv_src'], names=col_names)
        elif 'np_array' in kwargs:
            df = pd.DataFrame(data=kwargs['np_array'], 
                index=None, columns=col_names)
        df.set_index('BEPT', inplace=True)
        self.df = pd.merge(self.df, df, how='left',
            left_index=True, right_index=True)
        return




def main():
    wayne = PredictLable(csv_src='./Wayne_rd_intersections.txt')
    # wayne.load_column(csv_src='./predict_median.csv', col_name='median')
    wayne.load_column(np_array=np.load('predict_median.npy'), col_name='median')
    return

if __name__ == "__main__":
    main()