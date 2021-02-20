import numpy as np
import scipy as sp
import pandas as pd
# from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from scipy.stats import norm


#%%
def data_visual(data_df, col, bns = 100):
   '''

   :param data_df:  датафрейм
   :param col:      имя столбца с данными
   :param bns:      количество интервалов для гистограммы 100 по умолчанию
   :return:         None
   '''

   _, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(19,5))

   data_tst = data_df[col]
   # Шапиро тест на нормальность распределения
   print('Шапиро-тест переменной {}:\n {}'.format( col, sp.stats.shapiro(data_tst)))
   ## Общий заголовок
   _.suptitle('Анализ переменной:'+ col,y = 1.02,fontsize = 20)

   ## q-q plot тест на нормальность
   sp.stats.probplot(data_tst, plot=axes[0])
   axes[0].set_title('Тест на нормальное распределение \n' + col)
   axes[0].get_lines()[0].set_marker('o')
   axes[0].get_lines()[0].set_color('black')
   axes[0].get_lines()[0].set_markerfacecolor('lightblue')
   axes[0].get_lines()[0].set_markersize(8.0)

   ## Гистограмма
   axes[1].set_title('Распределение \n' + col)
   sns.distplot(data_tst, bins = bns, ax = axes[1])
   # Линии процентилей
   axes[1].axvline(x=np.percentile(data_tst, 25), color="r", linestyle="--", label = "25%")
   axes[1].axvline(x=np.percentile(data_tst, 50), color="g", linestyle="--", label = "50%")
   axes[1].axvline(x=np.percentile(data_tst, 75), color="r", linestyle="--", label = "75%")
   axes[1].axvline(x=np.percentile(data_tst, 99), color="b", linestyle="--", label="99%")
   # отобразим среднее
   axes[1].axvline(x=data_tst.mean(), color="r", linestyle="-", label = "mean")
   axes[1].legend()
   ## BOXPLOT
   axes[2].set_title('Boxplot\n' + col)
   sns.boxplot(data_tst,  ax = axes[2], orient="v", color = "moccasin")
   axes[2].set_xlabel(col)
   plt.show()



#%%
# вычисление статистик
def dict_statistic(data_df=None, stat = None, perc_1 = None, perc_2 = None, moda = False, n = 1, nan_info = False):
    '''

    :param data_df:     датафрейм
    :param stat:        статистика
    :param perc_1:      начальный перцентиль
    :param perc_2:      конечный перцентиль
    :param moda:        признак вычисления моды
    :param n:           Количество значений самых популярных значений (множестовенные моды)
    :param nan_info:    Признак вычисления количества NAN в данных
    :return:            Первый элемент словарь, второй датафрейм(мода)
    '''

    # служебная функция для имени моды
    def mode_text(s): #
        s ="mode_" + s
        return s

   # Основная функция
    value = []
    dict_data = {}
    main_dict = {}
    res_df = None
    df_cols=  data_df.describe().columns       # колонки по базовым статистикам

    # если считаем NaN
    if nan_info:

        res_df = data_df.isna().sum().to_frame().transpose()   # посчитали Nan
        filter_lst = df_cols.tolist()                          # колонки по базовым статистикам
        filter_lst.insert(0, 'NaN_count')                      #  добавили наименование показателя
        res_df = res_df.reindex(columns = filter_lst)          # отфильтрованные столбцы
        res_df.loc[0,'NaN_count'] = 'NaN_count'                # записали значение в новый столбец
        res_df = res_df.set_index('NaN_count').rename_axis('') # сделали индексом столбец и убрали имя

    # если выбрана мода
    if moda:

        for col in df_cols:
            buf = [] # для набора значений в колонке
            column_dict =  dict(data_df[col].value_counts().nlargest(n))  # набор  значений мод

            for i in column_dict:
                buf_1 = {i: column_dict[i]} # для  одного значения из набора
                buf.append(buf_1)

            main_dict[col] = buf   # готовый столбец датафрейма записали

        res_df = pd.DataFrame(main_dict) # готовый датафрейм
        res_df['tt'] = res_df.index.values.astype('str') # создаем текстовое имя
        res_df['tt']  = res_df.apply(lambda xx: mode_text(xx.tt), axis = 1) # составное текстовое имя
        res_df = res_df.set_index('tt').rename_axis('') # убираем служебное имя у индекса датафрейма

    # если считаем статистики
    if stat is not None :

        if (perc_1 is not None ) & (perc_2 is not None ) :# задан диапазон  перцентилей
            for col in df_cols:
                value.append(stat(data_df[col], perc_2) - stat(data_df[col], perc_1))

        if (perc_1 is not None ) & (perc_2 is None ): # задан один перцентиль
            for col in df_cols:
                value.append(stat(data_df[col], perc_1))

        if (perc_1 is None ) & (perc_2 is None ):  # перцентилей как параметров нет
            for col in df_cols:
                value.append(stat(data_df[col]))

        # Результат : словарь с показателем
        dict_data = dict(zip(list(df_cols), value))

    return  dict_data, res_df # первый  аргумент словарь, второй датафрейм(мода)

#%%
# ВЫДЕЛЕНИЕ ТИПОВ(украшательство)
def highlight_type(row, col, look_style = 1):
    '''

    :param row:         строка
    :param col:         столбец
    :param look_style:  1 красить один столбец 2 красить строку
    :return:            стиль окраски
    '''

    # палитра цветов
    color = {'A':'background-color: mistyrose','B':'background-color: lightcyan',
             'C':'background-color: moccasin', 'D':'background-color: lightyellow', 'F':'background-color: lavender'}

    has_style = row.apply(lambda i: '')

    if look_style == 1:
        column_index = row.index.tolist().index(col)
        has_style[[column_index]] =  color[row[col][0]]  # отдельные столбцы

    if look_style == 2:
        has_style[row.index.tolist()] =  color[row[col][0]]  # выделим  всю строку

    return has_style

#%%
# Функция для описания  данных
def data_describe(data_df, style = 1):
    '''

    :param data_df: датафрейм с переменными
    :param style:   1 красить столбец, 2 красить строку полностью
    :return:        1 стиль с таблице, 2 датафрейм с показателями
    '''

    # словарь показателей для сортировки в отчете
    metric_dic = {'count': 'A00' ,'mean': 'B01','std':'B02','min':'C01','25%':'C02','50%':'C03','75%':'C04','max':'C05','variance':'B03','IQR':'C07','kurtosis':'D01','skew':'D02','NaN_count':'A01', 'upper_limit':'C06', 'lower_limit':'C00'}
    res = data_df.describe()                                            # базовый отчет
    res.loc['variance']  = dict_statistic(data_df, np.var)[0]                # дисперсия
    res.loc['IQR'] = dict_statistic( data_df , np.percentile,25, 75)[0]      # межквартильное расстояние
    res.loc['kurtosis'] = dict_statistic( data_df , sp.stats.kurtosis)[0]       # эксцесс
    res.loc['skew'] = dict_statistic( data_df , sp.stats.skew)[0]               # ассиметрия
    res = pd.concat([res, dict_statistic(data_df, moda = True, n = 5)[1] ])  # моды  тк может быть бимодальность и тд
    res = pd.concat([res, dict_statistic(data_df, nan_info = True)[1] ])     # наличие  NAN

    # расчет дополнительных статистик
    # lower_limit = p_25 - IQR * 1.5
    # upper_limit = p_75 + IQR * 1.5

    df_IQR = pd.DataFrame.from_dict(dict_statistic( data_df , np.percentile,25, 75)[0], orient='index')
    df_75  = pd.DataFrame.from_dict(dict_statistic( data_df , np.percentile,75)[0], orient='index')
    df_25  = pd.DataFrame.from_dict(dict_statistic( data_df , np.percentile,25)[0], orient='index')
    df_upper_limit = df_75 + df_IQR * 1.5
    res.loc['upper_limit'] = df_upper_limit.to_dict()[0]
    df_lower_limit = df_25 - df_IQR * 1.5
    res.loc['lower_limit'] = df_lower_limit.to_dict()[0]

    # Добавление  колонки Type для сортировки показателей
    lst = res.columns.tolist()
    lst.insert(0, 'Type')     # помещение колонки в начало
    res = res.reindex(columns = lst)
    res.loc[:,'Type'] = 'F00' # заполнение колонки признаком

    # Проставим признаки согласно словаря
    for key in metric_dic:
        res.loc[key,'Type'] = metric_dic[key]
    res.sort_index(inplace=True)
    res = res.sort_values('Type', ascending= True)  # сортировка
    return res.style.apply(lambda xx: highlight_type(xx, 'Type', look_style = style ), axis=1), res


#%%
# График bootstrap
def boot_data_visual(boot_data, quants, name =''):
    '''

    :param boot_data:   столбец датафрейма
    :param quants:      датафрейм (индекс порядок квантиля столбец значение)
    :param name:        имя параметра
    :return:            строит график
    '''

    _, _, bars = plt.hist(boot_data, bins = 50)
    for bar in bars:
        if bar.get_x() <= quants.iloc[0][0] or bar.get_x() >= quants.iloc[1][0]:
            bar.set_facecolor('red') # красим за пределами доверительного интервала
        else:
            bar.set_facecolor('grey')
            bar.set_edgecolor('black')

    plt.style.use('ggplot')
    plt.vlines(quants,ymin=0,ymax=50,linestyle='--')
    plt.xlabel('boot_data')
    plt.ylabel('frequency')
    plt.title("Histogram of boot_data \n" + name)
    plt.show()


#%%
# Новая версия бутстрэп
def get_bootstrap_rev(
                        data_column_1 ,
                        data_column_2 ,
                        boot_it = 1000 ,
                        statistic = np.mean,
                        bootstrap_conf_level = 0.95 ,
                        visual = True ,
                        name = 'set',
                        timer = True
                    ):
    '''

    :param data_column_1:           числовые значения первой выборки
    :param data_column_2:           числовые значения второй выборки
    :param boot_it:                 количество бутстрэп-подвыборок
    :param statistic:               интересующая нас статистика
    :param bootstrap_conf_level:    уровень значимости
    :param visual:                  строим график или нет
    :param name:                    summary column
    :param timer:                   бегунок
    :return:                        {"boot_data": boot_data,"summary": inf}
    '''
    boot_len = max([len(data_column_1), len(data_column_2)]) # оставляем размер выборки внутри процедуры !!!
    boot_data = []
    for i in (tqdm(range(boot_it)) if timer else range(boot_it)): # извлекаем подвыборки и бегунок активируем
        samples_1 = data_column_1.sample(
            boot_len,
            replace = True # параметр возвращения значений выборки
        ).values

        samples_2 = data_column_2.sample(
            boot_len, # чтобы сохранить дисперсию, берем такой же размер выборки
            replace = True
        ).values

        boot_data.append(statistic(samples_1)-statistic(samples_2)) # разница

    pd_boot_data = pd.DataFrame(boot_data)  # пишем список в датафрейм

    left_quant = (1 - bootstrap_conf_level)/2 # уровень значимости 5% по умолчанию пополам
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = pd_boot_data.quantile([left_quant, right_quant]) # расчитали квантили интервала

    p_1 = norm.cdf(
        x = 0,
        loc = np.mean(boot_data),
        scale = np.std(boot_data)
    )


    p_2 = norm.cdf(
        x = 0,
        loc = -np.mean(boot_data),
        scale = np.std(boot_data)
    )


    p_value = min(p_1, p_2) * 2

    sign_result = 1 if ((1- bootstrap_conf_level) > p_value) else 0

    inf = { 'Significant Result': sign_result,
            'st_name': statistic.__name__,
            'confidence level':bootstrap_conf_level,
            'quants_level': [round(left_quant,4),round(right_quant,4)],
            'a1':  statistic(data_column_1),
            'a2':  statistic(data_column_2),
            'TST_diff_a1_a2':(statistic(data_column_1)- statistic(data_column_2)),
            'Conf_Int': list(round(quants[0],5)),
            '(H0: a1 = a2), p_value': p_value,
            '(H0: a1 > a2), p_value': p_2,
            '(H0: a1 < a2), p_value': p_1,
          }

    inf = pd.DataFrame(pd.Series(inf)).rename(columns = {0:name})

    if visual:
        boot_data_visual(pd_boot_data[0], quants, name =  data_column_1.name)

    result = {"boot_data": boot_data,
            "summary": inf}
    return (result)
#%%
# Процедура пакетной обработки
def package_bootstrap (a1, a2, column_list , st = np.mean, boot_it = 1000, visual = False):
    '''

    :param a1:          фрейм с переменными группы 1
    :param a2:          фрейм с переменными группы 2
    :param column_list: список обрабатываемых переменных
    :param st:          статистика по умолчанию np.mean
    :param boot_it:     количество выборок
    :param visual:      выводить график (да, нет)
    :return:            фрейм с отчетом
    '''
    empty_col = { 'Significant Result': 0,
            'st_name': 0,
            'confidence level': 0,
            'quants_level': 0,
            'a1': 0,
            'a2': 0,
            'TST_diff_a1_a2': 0,
            'Conf_Int': 0,
            '(H0: a1 = a2), p_value': 0,
            '(H0: a1 > a2), p_value': 0,
            '(H0: a1 < a2), p_value': 0,
          }

    res = pd.DataFrame(pd.Series(empty_col))

    for column in column_list:
        t1 = a1[column]
        t0 = a2[column]
        print(column)
        temp = get_bootstrap_rev(t1,
                                 t0,
                                 bootstrap_conf_level = 0.95,
                                 statistic = st,
                                 boot_it =  boot_it,
                                 visual = visual,
                                 name = column,
                                 timer = True)
        res = pd.concat([res, temp['summary']], axis=1, sort=False)

    res.drop(0, axis=1, inplace=True)
    res = np.transpose(res)
    return (res)



#%%
#Процедура описания матрицы связи
def class_describe(data_y=pd.Series(), data_x=pd.Series(),df=pd.DataFrame(), name_y='Actual', name_x='Predicted', name_title='Confusion matrix'):
    '''
    :param data_y:      пандас серия
    :param data_x:      пандас серия
    :param df:          датафрейм уже сформированного отчета например конфьюжен матрица (если да то первые параметры игнор)
    :param name_y:      подпись оси X
    :param name_x:      подпись оси Y
    :param name_title:  заголовок
    :return:            строит графики
    '''
    if df.empty:
        df = pd.crosstab(data_y, data_x,rownames=[name_y], colnames=[name_x],margins=True)

    plt.figure(figsize=(10, 12))
    fig, ax = plt.subplots()
    sns.heatmap(df, cmap ='viridis', annot= True,fmt='g',linewidths=.5)
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title(name_title,y=1.1 )
    plt.ylabel(name_y)
    plt.xlabel(name_x)
    plt.show()

    sns.heatmap(((df / df.iloc[-1,:]*100).iloc[:,: -1]).round(1), cmap ='viridis', annot= True, fmt='g',linewidths=.5)
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Пропорции параметра(%): ' + '"'+ name_x +'"',y=1.1 )
    plt.ylabel(name_y)
    plt.xlabel(name_x)
    plt.show()

    df = ((((df.T)/(df.iloc[:,-1]).T).T *100).iloc[: -1,:]).round(1)
    sns.heatmap(df, cmap ='viridis', annot= True, fmt='g',linewidths=.5)
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Пропорции параметра(%): ' + '"' + name_y + '"', y=1.1 )
    plt.ylabel(name_y)
    plt.xlabel(name_x)
    plt.show()

#%%
# Построение радара для двух наборов
import plotly.graph_objects as go

def radar_visual(st, data1, data2, name1, name2, up_limit, inf =''):
     '''

     :param st:          статистика 'mean' либо ' median'
     :param data1:       фрейм с параметрами 1
     :param data2:       фрейм с параметрами 2
     :param name1:       имя типа 1
     :param name2:       имя типа 2
     :param up_limit:    макс значение шкалы
     :param inf:         строка пояснение к заголовку
     :return:            строит график и возращает словарь {name1:data1,name2: data2}
     '''

     if st == 'mean':
         data1 = data1.mean().to_frame().reset_index().\
                              rename(columns={'index':'feature', 0:'r'})
         data2 = data2.mean().to_frame().reset_index().\
                              rename(columns={'index':'feature', 0:'r'})
     if st == 'median':
         data1 = data1.median().to_frame().reset_index().\
                              rename(columns={'index':'feature', 0:'r'})
         data2 = data2.median().to_frame().reset_index().\
                              rename(columns={'index':'feature', 0:'r'})

     cat_lst = list(data1.feature)
     data1_lst = list(data1.r)
     data2_lst = list(data2.r)

     fig = go.Figure()

     fig.add_trace(go.Scatterpolar(
         r= data1_lst,
         theta= cat_lst,
         fill='toself',
         name= name1
         ))
     fig.add_trace(go.Scatterpolar(
         r=data2_lst,
         theta=cat_lst,
         fill='toself',
         name=name2
         ))
     # fig.update_layout(legend_orientation="h")
     fig.update_layout(
        title="Сравнение по параметру " + st + inf,
        polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, up_limit]
        )),
        legend_orientation="h",
        legend=dict(x=.5, xanchor="center"),

        margin=dict(l=1, r=0, t=80, b=0),
        showlegend=True
        )

     fig.show()

     return ({name1:data1,name2: data2})

#%%
#
# # Изолирующий лес
# from sklearn.ensemble import IsolationForest
#
# def outliers_IsolForest(data, drop_lst, n_tree = 100, max_f = 1,cntm = 0.1 ):
#          X = data.drop(drop_lst, axis= 1)
#          labels = IsolationForest(random_state=123,
#                          n_estimators = n_tree,
#                          contamination =  cntm,
#                          max_features=max_f).fit_predict(X)
#          data['outliers'] = pd.Series(labels)
#          print("Количество выбросов:",(pd.Series(labels)== (-1)).sum())
#          return data
#%%

# Процедура сравнения двух наборов классов данных
def compare_pair(dat_df,cls_col, cls_lst, var_lst, name1='',name2=''):
    '''
    :param dat_df:  фрейм с переменными
    :param cls_col: имя колонки с переменной классификации
    :param cls_lst: список классов (выбираем два)
    :param var_lst: список переменных(столбцов) для сравнения
    :param name1:   имя первого класса
    :param name2:   имя второго класса
    :return:        строит график
    '''

    for item in  var_lst:

        df = dat_df[(dat_df[cls_col] == cls_lst[0])|(dat_df[cls_col] == cls_lst[1])]
        # df = data_all_trim.query('(new_cls =='+ str(cls_a) +') | (new_cls == '+str(cls_b)+')')

        _, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(15,3))
        _.suptitle('Переменная c указаним медианы :'+ item+ '\n' + cls_col+':'+
                   str(cls_lst[0]) +' -"'+name1+'", '+ cls_col+':'+
                   str(cls_lst[1]) +' -"'+name2+'"', y = 1.02,fontsize = 14)
        # сравнение пары по  одной переменной\

        sns.kdeplot(data=df,
                x=item,
                hue=cls_col,
                fill=True,
                common_norm=False,
                palette="crest",
                alpha=.5, linewidth=0,
                ax = axes[0]
                 )
        axes[0].axvline(x=df[df[cls_col]== cls_lst[0]][item].median(), color="g", linestyle="--", label = "median1")
        axes[0].axvline(x=df[df[cls_col]== cls_lst[1]][item].median(), color="b", linestyle="--", label = "median2")

        sns.boxplot(x=item,
                     y=cls_col,
                     orient="h",
                    data=df,
                    ax = axes[1],
                    palette="Set3",
                    )
        plt.show()

#%%
# процедура сравнения категорий
def categor_pair(df,cat_main, cat_x, name_main, name_x, lst_main=[],single = False ):
    '''

    :param df:          датафрейм
    :param cat_main:    имя столбца основной категории (легенда)
    :param cat_x:       категория по оси х
    :param name_main:   имя легенды
    :param name_x:      имя оси х
    :param lst_main:    список категорий легенды информационный
    :param single:      false для пары категорий в заголовке, true одна категория в заголовке
    :return:            строит график
    '''


    sns.set()
    plt.figure(figsize = (10,6))
    g = sns.countplot(hue=cat_main, x=cat_x, data=df)
    if len(lst_main)==0:
        g.legend(title=name_main, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    else:
        g.legend(title=name_main,labels = lst_main, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if single:
        plt.title('Категория:'+'\n'+'"'+ name_main+'"', fontsize = 20)
    else:
        plt.title('Связь категорий:'+'\n'+'"'+ name_main+'"'+ ' и  '+'"'+ name_x +'"', fontsize = 20)
    plt.ylabel("Количество", fontsize = 15)
    plt.xlabel(name_x, fontsize = 15)
    plt.show()
