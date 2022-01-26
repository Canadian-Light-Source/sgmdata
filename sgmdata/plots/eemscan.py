from bokeh.layouts import column, row, gridplot, layout
from bokeh.palettes import all_palettes
from bokeh.models import CustomJS, ColumnDataSource, Select, RangeSlider, ColorBar, LinearColorMapper, Rect, Button, \
    CheckboxButtonGroup, Slider, RadioGroup
from bokeh.plotting import Figure, output_notebook, output_file, show
from bokeh.embed import json_item
from bokeh import events
import bokeh
import json
import numpy as np
import os

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        output_notebook()
    # Jupyter notebook or qtconsole
    else:
        output_file("bokeh.html")  # Other type (?)
except NameError:
    output_file("bokeh.html")

required = ['image', 'sdd1', 'sdd2', 'sdd3', 'sdd4', 'emission', 'en', 'tey', 'pd', 'io', 'i0']
version = '1.x.x' if '1' in bokeh.__version__[0] or '0' in bokeh.__version__[0] else '2.x.x'


def get_callback(name):
    with open(os.path.dirname(os.path.abspath(__file__)) + '/callbacks/eemscan/' + name + f'_{version}.js', 'r') as f:
        js = f.read()
    return js

def plot(**kwargs):
    #Check vars
    sizing_mode = kwargs.get('sizing_mode', 'fixed')
    scale = kwargs.get('scale', 1)
    height, width = (int(600*scale), int(600*scale))
    if 'emission' not in kwargs.keys():
        kwargs['emission'] = np.linspace(0, 2560, 256)
    if 'io' in kwargs.keys() and np.any(kwargs['io']):
        kwargs['i0'] = kwargs['io']
    if "filename" not in kwargs.keys():
        filename = "xas"
    else:
        filename = kwargs['filename']

    delta = max(kwargs['en']) - min(kwargs['en'])
    bins = max(kwargs['emission']) - min(kwargs['emission'])

    #Data Sources
    source = ColumnDataSource(dict(image=[kwargs['image'].T],
                                   sdd1=[kwargs['sdd1'].T],
                                   sdd2=[kwargs['sdd2'].T],
                                   sdd3=[kwargs['sdd3'].T],
                                   sdd4=[kwargs['sdd4'].T],
                                   en=[min(kwargs['en'])],
                                   emission=[min(kwargs['emission'])],
                                   delta=[delta],
                                   bins=[bins]))
    xrf_source = ColumnDataSource(data=dict(proj_x=np.sum(source.data['image'][0], axis=1),
                                            emission=kwargs['emission'],
                                            proj_x_tot=np.sum(source.data['image'][0], axis=1),
                                            emission_tot=kwargs['emission'],
                                            sdd1=np.sum(source.data['sdd1'][0], axis=1),
                                            sdd2=np.sum(source.data['sdd2'][0], axis=1),
                                            sdd3=np.sum(source.data['sdd3'][0], axis=1),
                                            sdd4=np.sum(source.data['sdd4'][0], axis=1))
                                  )
    proj_y = np.sum(source.data['image'][0], axis=0)
    tey_max = np.amax(kwargs['tey'])
    pd_max = np.amax(kwargs['pd'])
    io_max = np.amax(kwargs['i0'])
    if tey_max == 0:
        tey_max = 1
    if pd_max == 0:
        pd_max = 1
    if io_max == 0:
        io_max = 1
    aux_source = ColumnDataSource(data=dict(en=kwargs['en'],
                                            tey=(kwargs['tey'] / tey_max) * np.amax(proj_y),
                                            pd=(kwargs['pd'] / pd_max) * np.amax(proj_y),
                                            i0=(kwargs['i0'] / io_max) * np.amax(proj_y)
                                            ))
    xas_source = ColumnDataSource(data=dict(proj_y=proj_y,
                                            en=kwargs['en'],
                                            en_tot=kwargs['en'],
                                            proj_y_tot=np.sum(source.data['image'][0], axis=0),
                                            ))
    xy_source = ColumnDataSource(data=dict(xaxis=[np.linspace(min(kwargs['en']), max(kwargs['en']), len(kwargs['en']))],
                                           yaxis=[kwargs['emission']]))
    rect_source = ColumnDataSource({'x': [], 'y': [], 'width': [], 'height': []})
    peak_source = ColumnDataSource({'x': [], 'y': [], 'width': [], 'height': []})


    #Plots & Glyphs
    plot = Figure(plot_width=width, plot_height=height, tools="box_select,save,box_zoom, wheel_zoom,hover,pan,reset")
    color_mapper = LinearColorMapper(palette="Spectral11", low=1, high=np.amax(kwargs['sdd1']))

    im = plot.image(image='image', y='emission', x='en', dh='bins', dw='delta', source=source,
                    palette="Spectral11")
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, border_line_color=None, location=(0, 0),
                         height=height*8//10, width=width*1//20)

    xrf = Figure(plot_width=width*27//64, plot_height=height, y_range=plot.y_range, tools="save,hover,box_zoom, pan",
                 title="XRF Projection")
    fluo = Rect(x='y', y='x', width='width', height='height', fill_alpha=0.1, line_color=None, fill_color='yellow')
    xrf.add_glyph(peak_source, fluo)
    xrf.circle('proj_x', 'emission', source=xrf_source, alpha=0.6)
    xrf.yaxis.visible = False
    xrf.xaxis.major_label_orientation = "vertical"

    xas = Figure(plot_width=width, plot_height=height*27//64, x_range=plot.x_range, tools="save,hover,box_zoom,wheel_zoom,pan",
                  title="XAS Projection")
    xas.line('en', 'proj_y', source=xas_source, line_color='purple', alpha=0.6, legend_label="EEMs")
    xas.line('en', 'tey', source=aux_source, line_color='black', alpha=0.6, legend_label="TEY")
    xas.line('en', 'pd', source=aux_source, line_color="navy", alpha=0.6, legend_label="Diode")
    xas.legend.click_policy = "hide"
    xas.legend.location = "top_left"
    xas.legend.label_text_font_size = "8pt"
    xas.legend.background_fill_alpha = 0.0
    xas.xaxis.visible = False

    rect = Rect(x='x', y='y', width='width', height='height', fill_alpha=0.1,
                line_color='orange', fill_color='black')
    plot.add_glyph(rect_source, rect)
    plot.xaxis.axis_label = 'Incident Energy (eV)'
    plot.yaxis.axis_label = 'Emisison Energy (eV)'

    #Interactive plot widgets:
    select = CheckboxButtonGroup(name="Detector Select:", labels=['sdd1', 'sdd2', 'sdd3', 'sdd4'], active=[0],
                                 height=height*1//20, width=width*3//8)
    button = Button(label="Download XAS", button_type="success", height_policy="min", width_policy='min',
                    height=height*1//15, width=width*3//16)
    checkbox_group = RadioGroup(labels=["dx/dy", "1/y", "None"], active=2, name="Functions",  height_policy='min',
                                height=height*1//15, width=width*3//16)
    flslider = Slider(start=10, end=2560, value=1280, step=10, title="Line Peak",  height=height*1//20, width=width*3//16)
    wdslider = Slider(start=20, end=500, value=100, step=10, title="Line Width", height=height*1//20, width=width*3//16)
    slider = RangeSlider(title="Color Scale:", start=0, end=4 * np.amax(kwargs['sdd1']),
                         value=(0, np.amax(kwargs['sdd1'])), step=20, height=height*1//20, width=width*3//8)
    select_palette = Select(title="Colormap Select:", options=['Viridis', 'Spectral', 'Inferno'], value='Spectral',
                             height=height*1//25, width=width*3//8)

    #Declaring CustomJS Callbacks
    select_callback = CustomJS(args=dict(s1=source, xrf=xrf_source, xas=xas_source, xy=xy_source, sel=rect_source,
                                         flslider=flslider, wdslider=wdslider, alter=checkbox_group, det=select),
                               code=get_callback('select'))
    reset_callback = CustomJS(args=dict(s1=source,
                                        xrf=xrf_source,
                                        xas=xas_source,
                                        xy=xy_source,
                                        sel=rect_source,
                                        alter=checkbox_group,
                                        det=select,
                                        fluo=peak_source), code=get_callback('reset'))
    plot.add_layout(color_bar, 'left')
    det_select = CustomJS(args=dict(source=source, xrf=xrf_source), code=get_callback('det_select'))
    callback_color_palette = CustomJS(args=dict(im=im, cl=color_bar), code=get_callback('color_palette'))
    callback_color_range = CustomJS(args=dict(im=im, cl=color_bar), code=get_callback('color_range'))
    callback_flslider = CustomJS(args=dict(s1=source,
                                           xy=xy_source,
                                           fluo=peak_source,
                                           xrf=xrf_source,
                                           xas=xas_source,
                                           flslider=flslider,
                                           wdslider=wdslider,
                                           sel=rect_source,
                                           alter=checkbox_group
                                           ), code=get_callback('flslider'))
    download = CustomJS(args=dict(s2=xas_source, aux=aux_source, filename=f"{filename}.csv"), code=get_callback('download'))

    #Linking callbacks
    plot.js_on_event(events.SelectionGeometry, select_callback)
    plot.js_on_event(events.Reset, reset_callback)
    flslider.js_on_change('value', callback_flslider)
    wdslider.js_on_change('value', callback_flslider)
    checkbox_group.js_on_change('active', callback_flslider)
    slider.js_on_change('value', callback_color_range)
    select_palette.js_on_change('value', callback_color_palette)
    select.js_on_change('active', det_select, callback_flslider)
    button.js_on_event(events.ButtonClick, download)

    #Layout
    fluo = row(flslider, wdslider)
    functions = row(button, checkbox_group)
    options = column(select, functions, fluo, slider, select_palette)
    if sizing_mode == 'scale_both' or scale < 0.6:
        lout = layout([
            [xas],
            [plot, xrf, options]
        ], sizing_mode=sizing_mode)
    else:
        lout = gridplot([[xas, options], [plot, xrf]], sizing_mode=sizing_mode)
    if kwargs.get('json', False):
        return json.dumps(json_item(lout, "eems"))
    show(lout)
