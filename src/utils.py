import pandas as pd
import matplotlib.pyplot as plt      
        
def visualize_MA_with_std(df: pd.DataFrame,
                          window_size: list,
                          measure_id: int,
                          std_window: int = 0,
                          envelope_window: int = 48):
    '''
    주어진 데이터의 MA와 STD를 그린다. 
    Window Size List를 인자로 넘겨 해당 Size만큼의 MA와 STD를 계산한다. 
    '''
    plt.figure(figsize=(12, 3))
    temp = df[df.MeasureId == measure_id].reset_index(drop=True)
    temp.Scalar.plot(alpha=0.2)
    color_list = ['r', 'b', 'g', 'dodgerblue', 'violet', 'limegreen', 'aqua'] * 2
    for i, window in enumerate(window_size):
        temp.Scalar.rolling(window).mean().plot(label=('ma'+str(window)), color=color_list[i])
    if std_window != 0:
        temp.Scalar.rolling(std_window).std().plot(label=('std'+str(std_window)), color='k')
        temp.Scalar.rolling(std_window).var().plot(label=('var'+str(std_window)), color='k', alpha=0.4)
    if envelope_window != 0:
        temp.Scalar.rolling(envelope_window).max().rolling(int(envelope_window*24)).mean().plot(label=('max'+str(envelope_window)))
    plt.title(f'{measure_id}th Sensor')
    plt.legend(loc=2)
    plt.xlim(0)
    plt.show()
        
def rolling_processor(df:pd.DataFrame,
                      window_size:list, 
                      std_window:int = 0,
                      envelope_window:int=48):
    '''
    주어진 데이터의 MA와 STD 컬럼을 추가해준다
    '''

    df[f'ma{window_size}']=df.Scalar.rolling(window_size).mean()
    if std_window !=0:
        df[f'std{std_window}']=df.Scalar.rolling(std_window).std()
        df[f'var{std_window}']=df.Scalar.rolling(std_window).var()
    if envelope_window!=0:
        df[f'upper_envolope_ma{envelope_window}']=df.Scalar.rolling(envelope_window).max().rolling(envelope_window*24).mean()
    return df


def df_divider(df, list_of_measureid):
    '''
    df{measureid} 형태로 분리된 dataframe dict를 반환한다.
    '''
    result={}
    df.DateTime = pd.to_datetime(df.DateTime)
    for measureid in list_of_measureid:
        exec(f'df{measureid}=df[df.MeasureId=={measureid}].reset_index(drop=True)')
        exec(f'result[{measureid}]=(df{measureid})')
    
    return result



def not_operating_condition_exterminator(df, outlier_of_10000, outlier_of_12000):
    '''
    데이터의 inhomogeneousity로 인해 모든 데이터를 한번에 이상치 제거하기가 어려움. 
    10000줄 정도의 데이터와 12000줄 정도의 데이터가 기계가 작동 안한 index가 다르다. 
    
    std가 0 근처로 떨어질 때가 기계작동이 멈춘 상태라고 판단한다. 
    outlier index는 함수 외부에서 인자로 전달한다. 
    '''
    if len(df) > 12000:
        df=df.drop(outlier_of_12000).reset_index(drop=True)
    else:
        df=df.drop(outlier_of_10000).reset_index(drop=True)
    return df

def normalizer(df, window=1000):
    '''정규화'''
    mean=df.Scalar[:window].mean()
    std=df.Scalar[:window].std()
    df.Scalar=(df.Scalar-mean)/std
    return df

def data_cutter(df, start_point):
    return df.iloc[start_point:,:]