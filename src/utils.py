import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from keras import backend as K
import os
import numpy as np
import pandas as pd
import seaborn as sns

CUR_DIR = os.path.abspath(os.curdir)
ROOT_DIR = os.path.dirname(CUR_DIR)
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
EVAL_DIR = os.path.join(ROOT_DIR, "evaluation")
MODEL_PERF_DIR = os.path.join(EVAL_DIR, "model_performance")
GRAPHS_DIR = os.path.join(EVAL_DIR, "graphs")
writepath = os.path.join(MODEL_PERF_DIR, "performance.csv")

plt.style.use('ggplot')

def plot_loss(history,model):
    """
    The purpose of this function is to plot the validation and training loss function across epochs.
    """
    plt.plot(history.history['mae'], label='training')
    plt.plot(history.history['val_mae'], label='val')
    plt.xlabel('epoch')
    plt.ylabel('mae')
    plt.title(f'Loss for {model.name}')
    plt.legend(loc='upper right')
    output_path = os.path.join(GRAPHS_DIR,f'Loss Plot {model.name}.png')
    plt.savefig(output_path)
    plt.show()
    print(output_path)
    
def relu_advanced(x):
    from keras import backend as K
    """The purpose of this function is the bound the output value of the network between 1 and 5 inclusively which matches the domain the stars get on the reviews."""
    return (K.relu(x, max_value=5))

def transpose_df(df,reset_index,prefix):
    if reset_index == False:
        out_df = df.groupby('star',as_index=False)['prediction'].mean().T
    elif reset_index == True:
        out_df = pd.DataFrame(df.groupby('star')['prediction'].skew()).reset_index().T
    new_header = out_df.iloc[0]
    new_header = [f'{prefix}_{int(i)}_Star' for i in new_header]
    new_header
    out_df = out_df[1:] #take the data less the header row
    out_df.columns = new_header
    return out_df

def write_performance(model,mae,writepath,eval_df):
    data = {
        'model_name':model.name,
        'mae':mae
    }
    grouped_eval_df = eval_df.groupby('star',as_index=False)['prediction'].mean()
    avg_prefix = 'Average_Prediction_for'
    skew_prefix = 'Prediction_Skewness_for'
    avg_df = transpose_df(eval_df,False,avg_prefix)
    skew_df = transpose_df(eval_df,True,skew_prefix)
    
    # grouped_eval_df.to_csv(os.path.join(DATA_DIR,'output','summary', f'{model.name}'))
    for col in avg_df.columns:
        data.update({col:avg_df[col][0]})
    for col in skew_df.columns:
        data.update({col:skew_df[col][0]})
    out_df = pd.DataFrame(data,index=[0])
    mode = 'a' if os.path.exists(writepath) else 'w'
    header = False if os.path.exists(writepath) else True
    out_df.to_csv(writepath, mode=mode, index=False, header=header)
    # print message
    print("Performance appended successfully.")
    
def plot_distributions(model,eval_df,field):
    i=0
    colors = ['black', 'midnightblue', 'darkgreen','mediumpurple','darkred']
    if field == 'nb_of_words':
        # bins = 20
        max_val = 100
        bin_field_name = f'binned_{field}'
        eval_df = eval_df[eval_df.nb_of_words<=max_val]
        bins = list(range(0,max_val,5))#[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,85,90,95,1]
        labels = bins[:-1] # list(range(0,len(bins)-1))
        eval_df[bin_field_name] = pd.cut(eval_df[field], bins=bins, labels=labels)
        # plt.text(x=0.5, y=0.94, s=f"Mean Absolute Error by Review Length Distribution for model: {model.name}", fontsize=18, ha="center")
        # plt.title(label = f"Mean Absolute Error by Review Length Distribution for model: {model.name}", fontdict = {'fontsize': 18},pad = 0.9) 
        eval_df.groupby(bin_field_name, as_index=False)['absolute_error'].mean()
            # subdf.hist(field, ax = ax, rwidth=0.9,color = colors[i],bins = bins)
            # i+=1
        # sns.set(rc = {'figure.figsize':(15,8)})
        b = sns.barplot(bin_field_name, 'absolute_error', data=eval_df, ci = False, color = colors[2])
        # plt.xlabel(field, fontsize=18)
        # plt.ylabel('Mean Absolute Error', fontsize=16)
        plt.gcf().set_size_inches(17, 9)
        b.axes.set_title(f"Mean Absolute Error by Review Length Distribution for model: {model.name}",fontsize=20)
        b.set_xlabel(field, fontsize=17)
        b.set_ylabel('Mean Absolute Error', fontsize=15)
        b.tick_params(labelsize=14) 
        
        # plt.figure(figsize=(40, 30))
    else:
        bins = 10
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
        fig.delaxes(axes[1][2])
        plt.text(x=0.5, y=0.94, s=f"Model Performance Distribution by Stars for model: {model.name}", fontsize=18, ha="center", transform=fig.transFigure)
        plt.subplots_adjust(hspace=0.95)
        for ax, (name, subdf) in zip(axes.flatten(), eval_df.groupby('star')):
            subdf.hist(field, ax = ax, rwidth=0.9,color = colors[i],bins = bins)
            i+=1
            ax.set_title(name)
            ax.set_xlabel(field)
            ax.set_ylabel('Count')
    # Generate histograms
    plt.savefig(os.path.join(GRAPHS_DIR,f'{field.capitalize()}_Distribution_{model.name}.png'))
    plt.show()
    
def performance_evaluation(X_test, y_test, model):
    y_pred = model.predict(X_test)
    print(f'The prediction values range between {min(y_pred)} and {max(y_pred)}')
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')
    eval_df = pd.merge(X_test, y_test, left_index=True, right_index=True)
    eval_df['prediction'] = y_pred.ravel()
    eval_df['absolute_error'] = (eval_df['prediction'] - eval_df['star']).abs()
    eval_df['nb_of_words'] = eval_df['review'].str.split().str.len()
    eval_df.to_excel(os.path.join(DATA_DIR,'output','scoring', f'{model.name}.xlsx'),index=False, encoding='utf-8')
    plot_distributions(model,eval_df,'prediction')
    plot_distributions(model,eval_df,'nb_of_words')
    write_performance(model,mae,writepath,eval_df)
    print('Done')