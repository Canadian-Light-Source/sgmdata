from bokeh.layouts import column, row, gridplot
from bokeh.palettes import all_palettes
from bokeh.models import CustomJS, ColumnDataSource, Select, RangeSlider, ColorBar, LinearColorMapper, Rect, Slider, \
    Range1d, DataTable, TableColumn, Button, TextAreaInput, SelectEditor, CellEditor, IntEditor
from bokeh.plotting import Figure, show
from bokeh.embed import json_item
from bokeh import events
import bokeh
from sgmdata.utilities.lib import scan_lib, elements
from sgmdata.xrffit import gaussians
import json
import numpy as np
import os

required = ['image', 'sdd1', 'sdd2', 'sdd3', 'sdd4', 'tey', 'xp', 'yp', 'emission']
version = '1.x.x' if '1' in bokeh.__version__[0] or '0' in bokeh.__version__[0] else '2.x.x'


def get_callback(name):
    with open(os.path.dirname(os.path.abspath(__file__)) + '/callbacks/xrfmap/' + name + f'_{version}.js', 'r') as f:
        js = f.read()
    return js

def make_data(df, keys, sgm_data):
    data = {k: df.filter(regex=("%s.*" % k), axis=1).to_numpy() for k in keys}
    data.update({k: np.reshape(v, (len(df.index.levels[0]), len(df.index.levels[1]), v.shape[-1])) if len(
        v.shape) == 2 else np.reshape(v, (len(df.index.levels[0]), len(df.index.levels[1]))) for k, v in data.items()})
    data.update({n: df.index.levels[i] for i, n in enumerate(list(df.index.names))})
    data.update({'emission': sgm_data.scans["2020-01-31t15-01-25-0600"].entry1.fit['emission']})
    data.update({'peaks': sgm_data.scans["2020-01-31t15-01-25-0600"].entry1.fit['peaks']})
    data.update({'width': sgm_data.scans["2020-01-31t15-01-25-0600"].entry1.fit['width']})
    return data


def plot(**kwargs):
    sizing_mode = kwargs.get('sizing_mode', 'stretch_both')
    # Create datasources
    if 'emission' not in kwargs.keys():
        kwargs['emission'] = np.linspace(0, 2560, 256)
    length = kwargs['sdd1'].shape[0] * kwargs['sdd1'].shape[1]
    if 'peaks' in kwargs.keys() and 'width' in kwargs.keys():
        mca_len = kwargs['sdd1'].shape[2]
        mca_source = ColumnDataSource(dict(
            sdd1=[kwargs['sdd1'][:, :, i].T for i in range(0, mca_len)],
            sdd2=[kwargs['sdd2'][:, :, i].T for i in range(0, mca_len)],
            sdd3=[kwargs['sdd3'][:, :, i].T for i in range(0, mca_len)],
            sdd4=[kwargs['sdd4'][:, :, i].T for i in range(0, mca_len)]))
        ##Create XRF data source.
        xrf_source = ColumnDataSource(dict(
            emission=kwargs['emission'],
            x1=gaussians(kwargs['emission'], *np.sum(kwargs['sdd1'], axis=(0, 1)), width=kwargs['width'],
                         centre=kwargs['peaks']),
            x2=gaussians(kwargs['emission'], *np.sum(kwargs['sdd2'], axis=(0, 1)), width=kwargs['width'],
                         centre=kwargs['peaks']),
            x3=gaussians(kwargs['emission'], *np.sum(kwargs['sdd3'], axis=(0, 1)), width=kwargs['width'],
                         centre=kwargs['peaks']),
            x4=gaussians(kwargs['emission'], *np.sum(kwargs['sdd4'], axis=(0, 1)), width=kwargs['width'],
                         centre=kwargs['peaks'])

        ))
    else:
        ##Create XRF data source.
        xrf_source = ColumnDataSource(dict(
            emission=kwargs['emission'],
            x1=np.sum(kwargs['sdd1'], axis=(0, 1)),
            x2=np.sum(kwargs['sdd2'], axis=(0, 1)),
            x3=np.sum(kwargs['sdd3'], axis=(0, 1)),
            x4=np.sum(kwargs['sdd4'], axis=(0, 1)),

        ))

    if 'roi_start' in kwargs.keys() and 'roi_end' in kwargs.keys():
        roi_start = kwargs['roi_start']
        roi_end = kwargs['roi_stop']
        max_var = 0

        ##Sum images for ROI
        im1 = np.sum(kwargs['sdd1'][:, :, roi_start:roi_end], axis=2)
        im2 = np.sum(kwargs['sdd2'][:, :, roi_start:roi_end], axis=2)
        im3 = np.sum(kwargs['sdd3'][:, :, roi_start:roi_end], axis=2)
        im4 = np.sum(kwargs['sdd4'][:, :, roi_start:roi_end], axis=2)
    else:
        ##Find ROI centred around max variance.
        var = np.var(kwargs['sdd3'], axis=(0, 1))
        max_var = np.mean(np.where(var == np.amax(var))).astype(int)
        if max_var > 5:
            roi_start = max_var - 5
        else:
            roi_start = 0
        if max_var < 250:
            roi_stop = max_var + 5
        else:
            roi_stop = 255

        ##Get initial images at max variance
        im1 = kwargs['sdd1'][:, :, max_var]
        im2 = kwargs['sdd2'][:, :, max_var]
        im3 = kwargs['sdd3'][:, :, max_var]
        im4 = kwargs['sdd4'][:, :, max_var]

    img_source = ColumnDataSource(dict(image=[im1.T],
                                       im1=[im1.T],
                                       im2=[im2.T],
                                       im3=[im3.T],
                                       im4=[im4.T],
                                       im5=[kwargs['tey']]
                                       ))

    # Create main image plot
    x_delta = max(kwargs['xp']) - min(kwargs['xp'])
    y_delta = max(kwargs['yp']) - min(kwargs['yp'])
    plot = Figure(plot_width=600, plot_height=600, tools="box_select,save,box_zoom, wheel_zoom,hover,pan,reset")
    color_mapper = LinearColorMapper(palette="Spectral11", low=0, high=np.amax(im1))
    im = plot.image(image='image', y=min(kwargs['yp']), x=min(kwargs['xp']), dh=y_delta, dw=x_delta, source=img_source,
                    palette="Spectral11")

    ##add image plot annotations
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, border_line_color=None, location=(0, 0))
    plot.xaxis.axis_label = 'X (mm)'
    plot.yaxis.axis_label = 'Y (mm)'
    plot.add_layout(color_bar, 'left')

    # Create XRF plot
    xrf = Figure(plot_width=300, plot_height=250, tools="save,hover", title="XRF Projection")
    xrf.line('emission', 'x1', source=xrf_source, line_color='purple', alpha=0.6, legend_label="sdd1")
    xrf.line('emission', 'x2', source=xrf_source, line_color='blue', alpha=0.6, legend_label="sdd2")
    xrf.line('emission', 'x3', source=xrf_source, line_color='black', alpha=0.6, legend_label="sdd3")
    xrf.line('emission', 'x4', source=xrf_source, line_color='red', alpha=0.6, legend_label="sdd4")

    ##add xrf plot annotations
    xrf.xaxis.axis_label = 'Emission (eV)'
    xrf.yaxis.axis_label = 'Intensity (a.u.)'
    xrf.yaxis.visible = False
    xrf.legend.click_policy = "hide"

    ##Change Detector Source for image
    det_callback = CustomJS(args=dict(source=img_source), code=get_callback('det_select'))
    det_select = Select(title="Detector Select:", options=['sdd1', 'sdd2', 'sdd3', 'sdd4', 'tey'], value='sdd1')
    det_select.js_on_change('value', det_callback)

    ##Color Palette Change
    callback_color_palette = CustomJS(args=dict(im=im, cl=color_bar), code=get_callback('color_palette'))

    ##Color Intensity Change Callback
    callback_color_range = CustomJS(args=dict(im=im, cl=color_bar), code=get_callback('color_range'))

    ##Change Pallette Selectbox
    palette_select = Select(title="Colormap Select:", options=['Viridis', 'Spectral', 'Inferno'], value='Spectral')
    palette_select.js_on_change('value', callback_color_palette)
    ##Change Color Intensity Slider
    intensity_slider = RangeSlider(title="Color Scale:", start=0, end=2 * np.amax(im1),
                                   value=(0, np.amax(im1)), step=20, )
    intensity_slider.js_on_change('value', callback_color_range)

    if 'peaks' in kwargs.keys() and 'width' in kwargs.keys():
        ##ROI change
        callback_roi_select = CustomJS(args=dict(source=img_source, mca=mca_source, det=det_select), code=get_callback('roi_select'))
        roi_menu = [(i, "%.1f" % e) for i, e in enumerate(kwargs['peaks'])]
        roi_slider = Select(title="Fluorescence Line:", options=roi_menu, value="%.1f" % kwargs['peaks'][max_var])
        roi_slider.js_on_change('value', callback_roi_select)

        ##Layout and display
        options = column(det_select, intensity_slider, palette_select, xrf, roi_slider)
    else:
        options = column(det_select, intensity_slider, palette_select, xrf)

    layout = gridplot([[plot, options]], sizing_mode=sizing_mode)
    if kwargs.get('json', False):
        return json.dumps(json_item(layout, "xrf"))
    show(layout)


def plot_interp(**kwargs):
    """
    Function to plot interactive XRF maps from raw or interpolated sgm data.
        Keywords:
            **kwargs (dict):  DataDict from plot function.
    """
    sizing_mode = kwargs.get('sizing_mode', 'stretch_both')
    # Verify the data in kwargs
    if 'xp' in kwargs.keys() and 'yp' in kwargs.keys():
        x = kwargs['xp']
        y = kwargs['yp']
    else:
        raise (Exception, "Improper data passed to plot function. Need x & y axes")
    if 'sdd1' in kwargs.keys():
        # Reshape sdd arrays to image.
        sdd1 = np.reshape(kwargs['sdd1'], (len(x), len(y), kwargs['sdd1'].shape[-1]))
        sdd2 = np.reshape(kwargs['sdd2'], (len(x), len(y), kwargs['sdd2'].shape[-1]))
        sdd3 = np.reshape(kwargs['sdd3'], (len(x), len(y), kwargs['sdd3'].shape[-1]))
        sdd4 = np.reshape(kwargs['sdd4'], (len(x), len(y), kwargs['sdd4'].shape[-1]))
    else:
        raise (Exception, "Improper data passed to plot function. Need sdd signal")
    if 'tey' in kwargs.keys():
        tey = np.reshape(kwargs['tey'], (len(x), len(y))).T
    else:
        raise (Exception, "Didn't recieve tey signal.")

    # Colourbar / Mapper
    max_col = np.nanmax(np.add.reduceat(sdd1, np.arange(0, 256, 10), axis=-1))
    color_mapper = LinearColorMapper(palette="Viridis256", low=np.nanmin(sdd1),
                                     high=max_col)

    # Set the y and x axes range from actual data.
    yr = Range1d(start=max(y), end=min(y))
    xr = Range1d(start=max(x), end=min(x))

    # Data source for plot
    data = {'x': [max(x)],
            'y': [max(y)],
            'dw': [max(x) - min(x)],
            'dh': [max(y) - min(y)],
            'image': [np.sum(sdd1[:, :, 45:55], axis=2).T]
            }
    data.update({'tey': [np.nanmax(sdd1) * (tey / np.nanmax(tey))]})
    source = ColumnDataSource(data)

    # Data source for image bank of sdd signals
    sdd_data = {'sdd1': [sdd1[:, :, i].T for i in range(0, sdd1.shape[-1])],
                'sdd2': [sdd2[:, :, i].T for i in range(0, sdd2.shape[-1])],
                'sdd3': [sdd3[:, :, i].T for i in range(0, sdd3.shape[-1])],
                'sdd4': [sdd4[:, :, i].T for i in range(0, sdd4.shape[-1])],
                }
    sdd_source = ColumnDataSource(sdd_data)

    # Create XRF Map plot
    plot = Figure(plot_width=600,
                  plot_height=600,
                  tools="box_select,save,box_zoom,wheel_zoom,hover,pan,reset",
                  x_range=xr,
                  y_range=yr,
                  background_fill_color="black",
                  background_fill_alpha=1,
                  )

    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None
    im = plot.image(image='image', x='x', y='y', dw='dw', dh='dh', color_mapper=color_mapper,
                    source=source, name="xrf-plot")

    # add image plot annotations
    color_bar = ColorBar(color_mapper=color_mapper, border_line_color=None, location=(0, 0))
    plot.xaxis.axis_label = 'X (mm)'
    plot.yaxis.axis_label = 'Y (mm)'
    plot.add_layout(color_bar, 'left')

    # XRF Plot Data
    xrf_source = ColumnDataSource(dict(
        emission=kwargs.get('emission', np.linspace(0, kwargs['sdd1'].shape[-1] * 10, kwargs['sdd1'].shape[-1])),
        x1=np.sum(kwargs['sdd1'], axis=0),
        x2=np.sum(kwargs['sdd2'], axis=0),
        x3=np.sum(kwargs['sdd3'], axis=0),
        x4=np.sum(kwargs['sdd4'], axis=0),
    ))

    # Glyph to highlight XRF peak.
    ymax = np.max([np.amax(v) for k, v in xrf_source.data.items() if 'x' in k])
    halfmax = ymax / 2
    rect_source = ColumnDataSource({'x': [500], 'y': [halfmax], 'width': [100], 'height': [ymax]}, name='rectangle')
    rect = Rect(x='x', y='y', width='width', height='height', fill_alpha=0.1,
                line_color='orange', fill_color='black')

    # Create XRF plot
    xrf = Figure(plot_width=300, plot_height=250, tools="save,hover", title="Total XRF")
    xrf.line('emission', 'x1', source=xrf_source, line_color='purple', alpha=0.6, legend_label="sdd1")
    xrf.line('emission', 'x2', source=xrf_source, line_color='blue', alpha=0.6, legend_label="sdd2")
    xrf.line('emission', 'x3', source=xrf_source, line_color='black', alpha=0.6, legend_label="sdd3")
    xrf.line('emission', 'x4', source=xrf_source, line_color='red', alpha=0.6, legend_label="sdd4")
    xrf.add_glyph(rect_source, rect)

    ##add xrf plot annotations
    xrf.xaxis.axis_label = 'Emission (eV)'
    xrf.yaxis.axis_label = 'Intensity (a.u.)'
    xrf.yaxis.visible = False
    xrf.legend.click_policy = "hide"
    xrf.legend.background_fill_alpha = 0.6

    slider = RangeSlider(start=0, end=2560, step=10, value=(450, 550), title="Fluorescent Line: ")
    det_select = Select(title="Detector Select:", options=['sdd1', 'sdd2', 'sdd3', 'sdd4', 'tey'], value='sdd3')

    # Change Detector Source for image
    det_callback = CustomJS(args=dict(source=source,
                                      sl=slider,
                                      im=im,
                                      det=det_select,
                                      rect=rect_source,
                                      sdd=sdd_source), code=get_callback('det_sum_select'))
    det_select.js_on_change('value', det_callback)
    slider.js_on_change('value', det_callback)

    # Color Palette Change
    callback_color_palette = CustomJS(args=dict(im=im, cl=color_bar), code=get_callback('color_palette'))

    # Color Intensity Change Callback
    callback_color_range = CustomJS(args=dict(im=im, cl=color_bar), code=get_callback('color_range'))

    # Change Pallette Selectbox
    palette_select = Select(title="Colormap Select:", options=['Viridis', 'Spectral', 'Inferno'], value='Viridis')
    palette_select.js_on_change('value', callback_color_palette)
    # Change Color Intensity Slider
    color_max = np.max([np.amax(np.add.reduceat(x, np.arange(0, 256, 10), axis=-1)) for x in [sdd1, sdd2, sdd3, sdd4]])
    intensity_slider = RangeSlider(title="Color Scale:", start=0, end=2 * color_max,
                                   value=(0, max_col), step=20, )
    intensity_slider.js_on_change('value', callback_color_range)

    options = column(det_select, intensity_slider, palette_select, xrf, slider)

    layout = gridplot([[plot, options]], sizing_mode=sizing_mode)
    if kwargs.get('json', False):
        return json.dumps(json_item(layout, "xrf"))
    show(layout)


def plot_xyz(shift=False, table=False, **kwargs):
    """
    Function to plot interactive XRF maps from raw or interpolated sgm data.
        Keywords:
            shift (bool):  False (default) - compensates for cmesh x-variance if required.
            table (bool):  False (default) - displays helper tool / datatable for macro generation at SGM.
            **kwargs (dict):  DataDict from plot function.
    """
    sizing_mode = kwargs.get('sizing_mode', 'stretch_both')
    # Verify the data in kwargs
    if 'xp' in kwargs.keys() and 'yp' in kwargs.keys():
        x = kwargs['xp']
        y = kwargs['yp']
    else:
        raise (Exception, "Improper data passed to plot function. Need x & y axes")
    if 'sdd1' in kwargs.keys():
        sdd1 = np.add.reduceat(kwargs['sdd1'], np.arange(0, 256, 5), axis=1)
        sdd2 = np.add.reduceat(kwargs['sdd2'], np.arange(0, 256, 5), axis=1)
        sdd3 = np.add.reduceat(kwargs['sdd3'], np.arange(0, 256, 5), axis=1)
        sdd4 = np.add.reduceat(kwargs['sdd4'], np.arange(0, 256, 5), axis=1)
    else:
        raise (Exception, "Improper data passed to plot function. Need sdd signal")
    if 'command' in kwargs.keys():
        command = kwargs['command']
    else:
        raise (Exception, "Improper data passed to plot function. Need spec command argument")
    if 'tey' in kwargs.keys():
        tey = kwargs['tey']
    else:
        raise (Exception, "Didn't recieve tey signal.")

    # XRF Map source data
    n1 = ["sdd1-" + str(int(i / 5)) for i in range(0, 256, 5)]
    n2 = ["sdd2-" + str(int(i / 5)) for i in range(0, 256, 5)]
    n3 = ["sdd3-" + str(int(i / 5)) for i in range(0, 256, 5)]
    n4 = ["sdd4-" + str(int(i / 5)) for i in range(0, 256, 5)]

    z = sdd3[:, 15]
    color_mapper = LinearColorMapper(palette="Viridis256", low=min(z), high=max(z))
    xdelta = abs(float(command[2]) - float(command[3]))
    ydelta = abs(float(command[6]) - float(command[7]))
    height = ydelta / float(command[8])
    width = xdelta / (sdd3.shape[0] / float(command[8]))

    # Shift the X data to line up the rows at center.
    # Pre-process the x values. The data needs to be shifted due to how they were collected
    if shift:
        shift = 0.5
        shifted_data = np.zeros(len(x))
        shifted_data[0] = x[0]
        for i in range(1, len(x)):
            shifted_data[i] = x[i] + shift * (x[i] - x[i - 1])
        x = shifted_data

    # Set the y and x axes range from actual data.
    yr = Range1d(start=max(y), end=min(y))
    xr = Range1d(start=max(x), end=min(x))

    data = {'x': x, 'y': y, 'z': z}
    data.update({n: sdd1[:, i] for i, n in enumerate(n1)})
    data.update({n: sdd2[:, i] for i, n in enumerate(n2)})
    data.update({n: sdd3[:, i] for i, n in enumerate(n3)})
    data.update({n: sdd4[:, i] for i, n in enumerate(n4)})
    data.update({'tey': np.nanmax(data['sdd3-15']) * (tey / np.nanmax(tey))})
    source = ColumnDataSource(data)

    # XRF Coordinates to Clipboard.
    if table:
        # Create clipboard data source
        data = {'sample': [], 'coords': [], 'edges': [], 'nscans': [], 'type': []}
        clipboard_source = ColumnDataSource(data)
        columns = [
            TableColumn(field="sample", title="Sample ID"),
            TableColumn(field="type", title="Type", editor=SelectEditor(options=["sample", "reference"])),
            TableColumn(field="coords", title="Position (x,y)", editor=CellEditor()),
            TableColumn(field="edges", title="Edges", editor=SelectEditor(options=elements, )),
            TableColumn(field="nscans", title="Number of Scans (#)", editor=IntEditor(step=10)),
        ]
        data_table = DataTable(source=clipboard_source, columns=columns, editable=True, width=600, height=300,
                               name="macro-table")
        table_macro = Button(label="Make Macro", button_type="success")
        text_area = TextAreaInput(value="", rows=10)

        table_delete = Button(label="Delete Row", button_type="danger")

        # Create dictionary to pass to Spec Jupyter Client.
        macro_callback = CustomJS(args=dict(clip=clipboard_source, lib=scan_lib, tx=text_area, tbl=data_table),
                                  code=get_callback('macro_build'))

        # Callback to delete row of datatable
        delete_callback = CustomJS(args=dict(clip=clipboard_source), code=get_callback('delete_row'))

        table_delete.js_on_event(events.ButtonClick, delete_callback)
        table_macro.js_on_event(events.ButtonClick, macro_callback)

        # Callback to collect x,y position on tap and push to clipboard & datatable
        clipboard_callback = CustomJS(args=dict(clip=clipboard_source), code=get_callback('clipboard'))

    # Create XRF Map plot
    plot = Figure(plot_width=600,
                  plot_height=600,
                  tools="box_select,save,box_zoom,wheel_zoom,hover,pan,reset",
                  x_range=xr,
                  y_range=yr,
                  background_fill_color="black",
                  background_fill_alpha=1,

                  )

    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None
    if table:
        plot.js_on_event('tap', clipboard_callback)

    im = plot.rect(x='x', y='y', color={'field': 'z', 'transform': color_mapper}, width=width, height=height,
                   source=source, name="xrf-plot")

    # add image plot annotations
    color_bar = ColorBar(color_mapper=color_mapper, border_line_color=None, location=(0, 0))
    plot.xaxis.axis_label = 'X (mm)'
    plot.yaxis.axis_label = 'Y (mm)'
    plot.add_layout(color_bar, 'left')

    # XRF Plot Data
    xrf_source = ColumnDataSource(dict(
        emission=kwargs.get('emission', np.linspace(0, kwargs['sdd1'].shape[1] * 10, kwargs['sdd1'].shape[1])),
        x1=np.sum(kwargs['sdd1'], axis=0),
        x2=np.sum(kwargs['sdd2'], axis=0),
        x3=np.sum(kwargs['sdd3'], axis=0),
        x4=np.sum(kwargs['sdd4'], axis=0),
    ))

    # Glyph to highlight XRF peak.
    ymax = np.max([np.amax(v) for k, v in xrf_source.data.items() if 'x' in k])
    halfmax = ymax / 2
    rect_source = ColumnDataSource({'x': [765], 'y': [halfmax], 'width': [50], 'height': [ymax]}, name='rectangle')
    rect = Rect(x='x', y='y', width='width', height='height', fill_alpha=0.1,
                line_color='orange', fill_color='black')

    # Create XRF plot
    xrf = Figure(plot_width=300, plot_height=250, tools="save,hover", title="Total XRF")
    xrf.line('emission', 'x1', source=xrf_source, line_color='purple', alpha=0.6, legend_label="sdd1")
    xrf.line('emission', 'x2', source=xrf_source, line_color='blue', alpha=0.6, legend_label="sdd2")
    xrf.line('emission', 'x3', source=xrf_source, line_color='black', alpha=0.6, legend_label="sdd3")
    xrf.line('emission', 'x4', source=xrf_source, line_color='red', alpha=0.6, legend_label="sdd4")
    xrf.add_glyph(rect_source, rect)

    ##add xrf plot annotations
    xrf.xaxis.axis_label = 'Emission (eV)'
    xrf.yaxis.axis_label = 'Intensity (a.u.)'
    xrf.yaxis.visible = False
    xrf.legend.click_policy = "hide"
    xrf.legend.background_fill_alpha = 0.6

    slider = Slider(start=0, end=2560, step=51, value=765,
                    title="Fluorescent Line: ", width=300)
    det_select = Select(title="Detector Select:", options=['sdd1', 'sdd2', 'sdd3', 'sdd4', 'tey'], value='sdd3', width=300)

    # Change Detector Source for image
    det_callback = CustomJS(args=dict(source=source, sl=slider, im=im, det=det_select, rect=rect_source),
                            code=get_callback('det_select_xyz'))
    det_select.js_on_change('value', det_callback)
    slider.js_on_change('value', det_callback)

    # Color Palette Change
    callback_color_palette = CustomJS(args=dict(im=im, cl=color_bar), code=get_callback('color_palette'))

    # Color Intensity Change Callback
    callback_color_range = CustomJS(args=dict(im=im, cl=color_bar), code=get_callback('color_range'))

    # Change Pallette Selectbox
    palette_select = Select(title="Colormap Select:", options=['Viridis', 'Spectral', 'Inferno'], value='Viridis', width=300)
    palette_select.js_on_change('value', callback_color_palette)
    # Change Color Intensity Slider
    color_max = np.max([np.amax(x, axis=1) for x in [sdd1, sdd2, sdd3, sdd4]])
    intensity_slider = RangeSlider(title="Color Scale:", start=0, end=2 * color_max,
                                   value=(0, np.amax(z)), step=20, width=300)
    intensity_slider.js_on_change('value', callback_color_range)

    options = column(det_select, palette_select, intensity_slider, xrf, slider)
    if table:
        layout = gridplot([[plot, options],
                           [data_table, column(table_macro, table_delete, text_area)]])
    else:
        layout = gridplot([[plot, options]], sizing_mode=sizing_mode)
    if kwargs.get('json', False):
        return json.dumps(json_item(layout, "xrf"))
    show(layout)
