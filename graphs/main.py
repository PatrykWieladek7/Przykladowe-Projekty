"Example of using numpy,plotly and scikit-learn libraries, including ploting a linear Regression"
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import plotly.express as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
def data_1():
    "Create data where both arrays are numerical"
    x=np.random.randint(10, size=11)
    y=np.random.randint(8,size=11)
    return x, y
def data_2():
    "Create data where one of arrays consists of labels"
    x=np.array(["Apple","Orange","Pear","Apple","Apple","Pear","Apple","Pear","Orange","Apple","Apple"])
    y = np.random.randint(10, size=11)
    return x, y
def data_3():
    "Create data for linear regression"
    x=np.array(range(1,12))
    y=np.random.randint(8,size=11)
    return x, y
def graph_bar():
    "Example of bar graph"
    x,y=data_2()
    b=plt.bar(y,x,title="Example of Bar Graph",color_discrete_sequence=["purple"]*len(x))
    b.update_layout(font_color="red",xaxis_title="Fruit",yaxis_title="How many")
    return b
def graph_histogram():
    "Example of histogram graph"
    x,y=data_1()
    h=plt.histogram(x,y,title="Example of Histogram",color_discrete_sequence=["gray"]*len(x))
    h.update_layout(font_color="green",xaxis_title="Variable",yaxis_title="Result")
    return h
def graph_pie():
    "Example of pie graph"
    x,y=data_2()
    p=plt.pie(y,x)
    return p
def graph_Linear_Regression():
    x,y=data_1()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
    model=linear_model.LinearRegression(fit_intercept=True)
    model = model.fit(x[:, np.newaxis], y)
    y_pred= model.predict(x_test[:, np.newaxis])
    s=plt.scatter(x,y,title="Linear Regression example")
    s=s.add_traces(go.Scatter(x=x_test, y=y_pred, name='Predicted line'))
    s.update_layout(font_color="green", xaxis_title="Variable", yaxis_title="Result")
    return s
def apend_traces(a, graph,x,y):
    graph_traces = []
    for trace in range(len(graph["data"])):
        graph_traces.append(graph["data"][trace])
    for traces in graph_traces:
        a.append_trace(traces, row=x, col=y)
    return a

def all_subplots(b,h,l):
    "Creating subplots of diffrent graphs"
    a = make_subplots(rows=2, cols=2, start_cell="bottom-left")
    a=apend_traces(a,b,2,1)
    a=apend_traces(a,l,2,2)
    a=apend_traces(a,h,1,1)
    a.show()
if __name__ == '__main__':
    "Showcasing all graphs"
    l = graph_Linear_Regression()
    b = graph_bar()
    h = graph_histogram()
    p = graph_pie()
    b.show()
    h.show()
    l.show()
    p.show()
    all_subplots(b,l,h)


