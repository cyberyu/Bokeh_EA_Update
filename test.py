from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox
from bokeh.io import curdoc
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6
from bokeh.models import FactorRange, ColumnDataSource
from bokeh.models.widgets import Button

button = Button(label="ChangeValue", button_type="success")
fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
val = [5, 3, 4, 2, 4, 6]
data_dict = {'x':fruits,'y':val}
source_table_hist = ColumnDataSource(data=data_dict)

h = figure(x_range=data_dict['x'],plot_height=350, title="Histogram")
h.vbar(x='x', top='y', width=0.2, source=source_table_hist, legend="x", line_color='black',
       fill_color=factor_cmap('x', palette=Spectral6, factors=data_dict['x']))

h.xgrid.grid_line_color = None
h.y_range.start = 0

inputs = widgetbox(button)

def update():
    fruits = ['Banana', 'Orange']
    val = [15, 23]
    data_dict = {'x':fruits,'y':val}
    h.x_range.factors = data_dict['x']
    source_table_hist.data = data_dict

button.on_click(update)
l = layout([inputs,h])
curdoc().add_root(l)
curdoc().title = "Test"