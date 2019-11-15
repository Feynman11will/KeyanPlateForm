'''
@Author: your name
@Date: 2019-11-13 13:38:25
@LastEditTime: 2019-11-13 14:00:39
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /parse/streamlittutorial.py
'''
import streamlit as st
import pandas as pd
import numpy as np
import time
DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
        'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


def set_mardown():
    '''
    这本身就是一个在python里面的markdown文件
    '''
    st.title('1. streamlit 测试')

    st.write('如何显示markdown?使用write方法向markdown写文件')
    '显示表格'
    st.write(pd.DataFrame({
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40]
    }))
    
#################################################################################

def show_pbar():
    '''
    显示进度条
    '''

    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text('Iteration {}'.format(i+1))
        bar.progress(i + 1)
        time.sleep(0.1)
#################################################################################

#################################################################################


wonder_dict = {"hyperparameter":
     { "giou" : 1.582,
       "cls": 27.76,
       "cls_pw": 1.446,
       "obj": 21.35,
       "obj_pw": 3.941,
       "iou_t": 0.2635,
       "lr0": 0.00001,
       "lrf": -4.0,
       "momentum": 0.98,
       "weight_decay": 0.000004569,
       "fl_gamma": 0.5,
       "hsv_s": 0.5703,
       "hsv_v": 0.3174,
       "degrees": 1.113,
       "translate": 0.06797,
       "scale": 0.1059,
       "shear": 0.5768},
    "trainParameter":
        {"model":{"ds_path":"/data1/wanglonglong/FeiYan"},
         "n_epochs":10,
         "batch_size":8,
         "acumulate":2,
         "transfer":0,
         "img_size":640,
         "resume":1,
         "cache_images":1,
         "notest":0,
         "loss_func":"defaultpw",
         "optimizer":"adam",
         "augment":1,
         "device":""},
     "brightEnhancement":1,
     "spaceEnhancemenet":1,
     "classes":1,
     "inference":
         {"source":"/data1/wanglonglong/01workspace/yolov3_orig/yolov3-xray-chest/data/samples",
          "conf_thres": 0.5,
          "nms_thres":0.5,
          "iou_thres":0.5,
          "device":"0",
          "view_img":1,
          "test_result_path":"/data1/wanglonglong/01workspace/yolov3PtResult/FeiyanOk/backup30.pt"},
      "kmeans":1}


@st.cache
def show_json(wonder_dict=wonder_dict):
    '''
    展示一个jsoin文件
    '''
    return wonder_dict

#################################################################################

def show_mardwon(wonder_dict=wonder_dict):
    st.write('1. ergou')

    st.write('- ergou')


    '''
    pip install streamlit
    pip install -U altair vega_datasets

    ssh -o logLevel=ERROR -L 8501:$IP_ADDRESS:8501 $USERNAME@$IP_ADDRESS
    使用altair绘制图像是一个很好地选择
    写好一个脚本以后，使用
    也可以使用Plotly Bokeh Vega-Lite 等等
    '''

def show_slider():
    x= st.slider('x')
    st.write(x, 'squared is', x * x)

def show_anotation():
    
    '使用缓存机制'
    '''
    使用缓存机制，将信息缓存起来，维持app中的信息

    The actual bytecode that makes up the body of the function

    Code, variables, and files that the function depends on

    The input parameters that you called the function with

    1. function中的实际的bytecode
    2. 函数所依赖的代码变量和文件
    3. 传入的参数
    第一次运行程序，app会将结果缓存到local cache中，如果变量的值没有发生变化，则跳过执行代码，否则就从
    cache中读取变量

    - 使用限制
    1. cache机制仅仅检查python和python安装的环境
    2. 内存中的值必须确定
    3. 所有的值都是按照引用存储的
    '''


def show_sidebar():
    add_selectbox = st.sidebar.checkbox(
      'How would you like to be contacted?',
      ('Email', 'Home phone', 'Mobile phone'))


  # Adds a slider to the sidebar
    add_slider = st.sidebar.slider(
        'Select a range of values',
        0.0, 100.0, (25.0, 75.0))
    
def use_all():
    set_mardown()
    # show_pbar()
    st.write(show_json())
    show_mardwon()
    show_slider()
    show_anotation()
    show_sidebar()
def showmap():
    st.title('Uber pickups in NYC')
    data_load_state = st.text('Loading data...')
    data = load_data(10000)
    data_load_state.text('Loading data...done!')
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)
    
    st.subheader('Number of pickups by hour')
    hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
    st.bar_chart(hist_values)
    st.subheader('Map of all pickups')
    st.map(data)

    # # hour_to_filter = 17
    # st.sidebar()
    hour_to_filter = st.slider('hour', 0, 23, 17) 
    filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
    st.subheader('Map of all pickups at {}:00'.format(hour_to_filter))
    st.map(filtered_data)

def hyperUse():
    '''
    随机显示10*10的数字
    
        数据帧必须是二维的数据
    '''
    code = '''
    def showmap():
    st.title('Uber pickups in NYC')
    data_load_state = st.text('Loading data...')
    data = load_data(10000)
    data_load_state.text('Loading data...done!')
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)
    
    st.subheader('Number of pickups by hour')
    hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
    st.bar_chart(hist_values)
    st.subheader('Map of all pickups')
    st.map(data)

    # # hour_to_filter = 17
    # st.sidebar()
    hour_to_filter = st.slider('hour', 0, 23, 17) 
    filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
    st.subheader('Map of all pickups at {}:00'.format(hour_to_filter))
    st.map(filtered_data)
    '''

    st.code(code,language='python')
    # dataframe = np.random.randn(10, 20,10)
    dataframe = np.random.randn(10, 20)
    st.dataframe(dataframe)

    dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))
    '''这里我要好好学一学pandas数据可视化'''
    st.dataframe(dataframe.style.highlight_max(axis=0))

    '''
    使用st.table 绘制二维数据表格，显示到网页上来但是这种不支持网页折叠
    '''
    dataframe = pd.DataFrame(
        np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))
    st.table(dataframe)


    '''
    插入无序的交互场控件
    '''
    st.text('This will appear first')
    # Appends some text to the app.

    my_slot1 = st.empty()
    # Appends an empty slot to the app. We'll use this later.

    my_slot2 = st.empty()
    # Appends another empty slot.

    st.text('This will appear last')
    # Appends some more text to the app.

    my_slot1.text('This will appear second')
    # Replaces the first empty slot with a text string.

    my_slot2.line_chart(np.random.randn(20, 2))
    # Replaces the second empty slot with a chart.


def animate_elements():
    progress_bar = st.progress(0)
    # 使用占位符
    status_text = st.empty()

    chart = st.line_chart(np.random.randn(10, 2))

    for i in range(20):
        progress_bar.progress(i)
        new_rows=np.random.randn(10,2)
        status_text.text(
        'The latest random number is: %s' % new_rows[-1, 1])
        chart.add_rows(new_rows)
        time.sleep(0.1)
    status_text.text('Done! 显示小气球')
    st.balloons()
def api_show():
    if st.checkbox('api'):
        '- **显示mardown的内容**'
        st.markdown('Streamlit is **_really_ cool**.')

        '1. 还可以显示latex公式'

        st.latex(r'''
            a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
            \sum_{k=0}^{n-1} ar^k =
            a \left(\frac{1-r^{n}}{1-r}\right)
            ''')
        '2. 高级的write用法'
        if st.checkbox('''- st.write ha函数输出牛逼的代码结果;\
                        这里太长我就不展开看了'''):
            st.write(hyperUse())

        '''
        - write(string) : Prints the formatted Markdown string.
        - write(data_frame) : Displays the DataFrame as a table.
        - write(error) : Prints an exception specially.
        - write(func) : Displays information about a function.
        - write(module) : Displays information about the module.
        - write(dict) : Displays dict in an interactive widget.
        - write(obj) : The default is to print str(obj).
        - write(mpl_fig) : Displays a Matplotlib figure.
        - write(altair) : Displays an Altair chart.
        - write(keras) : Displays a Keras model.
        - write(graphviz) : Displays a Graphviz graph.
        - write(plotly_fig) : Displays a Plotly figure.
        - write(bokeh_fig) : Displays a Bokeh figure.
        - write(sympy_expr) : Prints SymPy expression using LaTeX.'''

        '''3. 输出公式'''
        a=2
        st.write(a)
        '''4. 使用altair画图看来很牛逼美观大方'''
        import altair as alt
        df = pd.DataFrame(
            np.random.randn(200, 3),
            columns=['a', 'b', 'c'])

        c = alt.Chart(df).mark_circle().encode(
            x='a', y='b', size='c', color='c')

        st.write(c)

        st.header('This is a header')
        st.subheader('This is a subheader')
        
        if st.checkbox('显示神奇的表格'):
            '''- this is so fucking niubi'''
            'st.area_chart'
            chart_data = pd.DataFrame(
                np.random.randn(20, 3),
                    columns=['a', 'b', 'c'])
            st.area_chart(chart_data)

            'st.bar_chart'
            chart_data = pd.DataFrame(
                np.random.randn(50, 3),
                columns=["a", "b", "c"])
            st.bar_chart(chart_data)

            import matplotlib.pyplot as plt

            arr = np.random.normal(1, 1, size=100)*100
            plt.hist(arr, bins=100)
            st.pyplot()

            import graphviz as graphviz

            st.graphviz_chart('''
                digraph {
                    run -> intr
                    intr -> runbl
                    runbl -> run
                    run -> kernel
                    kernel -> zombie
                    kernel -> sleep
                    kernel -> runmem
                    sleep -> swap
                    swap -> runswap
                    runswap -> new
                    runswap -> runmem
                    new -> runmem
                    sleep -> runmem
                }
            ''')
            from PIL import Image
            image = Image.open('测试图.jpg')
            st.image(image, caption='Sunrise by the mountains',
                 use_column_width=True)

def show_widget():
    agree = st.checkbox('I agree')

    if agree:
        st.write('Great!')

    st.markdown('1. **ergou**')

    genre = st.radio(
        '''**What's your favorite movie genre**''',
        ('Comedy', 'Drama', 'Documentary'))    
    if genre == 'Comedy':
        st.write('You selected comedy.')
    else:
        st.write('''You didn't select comedy.''')

    option = st.selectbox(
         'How would you like to be contacted?',
         ('Email', 'Home phone', 'Mobile phone'))
    st.write('You selected:', option)

    options = st.multiselect(
         'What are your favorite colors',('Green', 'Yellow', 'Red', 'Blue'))

    st.write('You selected:', options)
    '1. 文本输入'
    title = st.text_input('Movie title', 'Life of Brian')
    st.write('The current movie title is:', title)
    '2. 数字输入'
    number = st.number_input('Insert a number',0,22)
    st.write('The current number is ', number)
    '3. 多行文本输入'
    txt = st.text_area('Text to analyze', '''
        It was the best of times, it was the worst of times, it was
        the age of wisdom, it was the age of foolishness, it was
        the epoch of belief, it was the epoch of incredulity, it
        was the season of Light, it was the season of Darkness, it
        was the spring of hope, it was the winter of despair, (...)
        ''')
    st.write('Sentiment:', (txt))

    '4. 日期选择'
    import datetime
    d = st.date_input(
        'When s your birthday',
        datetime.date(2019, 7, 6))
    st.write('Your birthday is:', d)
    '5. 输入时间'
    t = st.time_input('Set an alarm for', datetime.time(8, 45))
    st.write('Alarm is set for', t)
    '6. '
    add_selectbox = st.sidebar.checkbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone'))
    ergou = st.sidebar.slider('ergou{}'.format(t),10,100)
    '7 . 代码展示'

    def get_user_name():
        return 'John'
    with st.echo():
        st.write('i am so tired')
    '''8 spiner是什么东西？\
    Temporarily displays a message while executing a block of code.'''
    with st.spinner('Wait for it...'):
        time.sleep(0.5)
        
    st.success('Done!')

    '''
    9.显示气球
    '''
    st.balloons()
    '10. 这是个错误'
    error = st.error('This is an error')
    st.write(error)
    '11. 显示了错误以后还可以运行吗'
    st.info('This is a purely informational message')
    '依然可以运行'
    '12 . 显示成功的消息'
    st.success('This is a success message!')

    '13. 使用占位符'
    my_placeholder = st.empty()
    my_placeholder.slider('滑动条', 0,100)
    my_placeholder.text('太累了')
    time.sleep(2)
    '''
    14 显示帮助文档
    '''
    my_placeholder.help(pd.DataFrame)

if __name__=='__main__':
    if st.checkbox('Show map example'):
        showmap()
    if st.checkbox('显示各种用法'):
        use_all()
    if st.checkbox('显示高级用法'):
        hyperUse()
    ##dataframe 
    if st.checkbox('显示动态元素'):
        animate_elements()
    api_show()

    if st.checkbox('显示交互式空间'):
        show_widget()


    COLOR='red'
    BACKGROUND_COLOR = '#DDDDFF'
    st.markdown(
                f"""
        <style>
            .reportview-container .main {{
                color: {COLOR};
                background-color: {BACKGROUND_COLOR};
            }}
        </style>
        """,
                unsafe_allow_html=True,
            )
    

        