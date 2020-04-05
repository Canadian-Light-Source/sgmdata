from bokeh.layouts import column, row, gridplot
from bokeh.palettes import all_palettes
from bokeh.models import CustomJS, ColumnDataSource, Select, RangeSlider, ColorBar, LinearColorMapper, Rect
from bokeh.plotting import Figure, output_notebook, output_file, show
from bokeh.embed import json_item
from bokeh import events
import numpy as np

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        output_notebook()
    # Jupyter notebook or qtconsole
    else:
        output_file("bokeh.html")  # Other type (?)
except NameError:
    output_file("bokeh.html")

required = ['image', 'sdd1', 'sdd2', 'sdd3', 'sdd4', 'xp', 'yp', 'emission']

def make_data(df, keys):
    data = {k: df.filter(regex=("%s.*" % k), axis=1).to_numpy()for k in keys}
    data.update({k:np.reshape(v, (len(df.index.levels[0]),len(df.index.levels[1]),v.shape[-1])) if len(v.shape) == 2 else np.reshape(v,(len(df.index.levels[0]),len(df.index.levels[1]))) for k,v in data.items()})
    data.update({n:df.index.levels[i] for i,n in enumerate(list(df.index.names))})
    return data

def plot(**kwargs):
    # Create datasources
    if 'emission' not in kwargs.keys():
        kwargs['emission'] = np.linspace(0, 2560, 256)
    mca_source = ColumnDataSource(dict(
        sdd1=[kwargs['sdd1']],
        sdd2=[kwargs['sdd2']],
        sdd3=[kwargs['sdd3']],
        sdd4=[kwargs['sdd4']]))

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
    im1 = np.sum(kwargs['sdd1'][:, :, roi_start:roi_stop], axis=2)
    im2 = np.sum(kwargs['sdd2'][:, :, roi_start:roi_stop], axis=2)
    im3 = np.sum(kwargs['sdd3'][:, :, roi_start:roi_stop], axis=2)
    im4 = np.sum(kwargs['sdd4'][:, :, roi_start:roi_stop], axis=2)

    img_source = ColumnDataSource(dict(image=[im1.T],
                                       im1=[im1.T],
                                       im2=[im2.T],
                                       im3=[im3.T],
                                       im4=[im4.T]
                                       ))

    ##Create XRF data source.
    xrf_source = ColumnDataSource(dict(
        emission=kwargs['emission'],
        x1=np.sum(kwargs['sdd1'], axis=(0, 1)),
        x2=np.sum(kwargs['sdd2'], axis=(0, 1)),
        x3=np.sum(kwargs['sdd3'], axis=(0, 1)),
        x4=np.sum(kwargs['sdd4'], axis=(0, 1)),

    ))

    # Create main image plot
    x_delta = max(kwargs['xp']) - min(kwargs['xp'])
    y_delta = max(kwargs['yp']) - min(kwargs['yp'])
    plot = Figure(plot_width=600, plot_height=600, tools="box_select,save,box_zoom, wheel_zoom,hover,pan,reset")
    color_mapper = LinearColorMapper(palette="Spectral11", low=1, high=np.amax(im1))
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
    det_callback = CustomJS(args=dict(source=img_source), code="""
            var sdd1 = source.data['im1'][0];
            var sdd2 = source.data['im2'][0];
            var sdd3 = source.data['im3'][0];
            var sdd4 = source.data['im4'][0];
            var d = source.data['image'];
            var f = cb_obj.value;
            if (f == "sdd1") {
                d[0] = sdd1;
            }
            if (f == "sdd2") {
                d[0] = sdd2;
            }
            if (f == "sdd3") {
                d[0] = sdd3;
            }
            if (f == "sdd4") {
                d[0] = sdd4;
            }
            source.change.emit();
            xrf.change.emit();
    """)
    det_select = Select(title="Detector Select:", options=['sdd1', 'sdd2', 'sdd3', 'sdd4'], value='sdd1')
    det_select.js_on_change('value', det_callback)

    # Color Palettes
    viridis = all_palettes['Viridis'][256]
    inferno = all_palettes['Inferno'][256]
    spectral = all_palettes['Spectral'][11]
    colorblind = all_palettes['Colorblind'][4]

    ##Color Palette Change
    callback_color_palette = CustomJS(args=dict(im=im, cl=color_bar), code="""
            var p = "Inferno11";
            var f = cb_obj.value;
            if (f == "Viridis") {
                im.glyph.color_mapper.palette = %s;
                cl.color_mapper.palette = %s;
            }
            if (f == "Spectral") {
                im.glyph.color_mapper.palette = %s;
                cl.color_mapper.palette = %s;
            }
            if (f == "Inferno") {
                im.glyph.color_mapper.palette = %s;
                cl.color_mapper.palette = %s;
            }
            if (f == "Colorblind") {
                im.glyph.color_mapper.palette = %s;
                cl.color_mapper.palette = %s;
            }
    """ % (viridis, viridis, spectral, spectral, inferno, inferno, colorblind, colorblind))

    ##Color Intensity Change Callback
    callback_color_range = CustomJS(args=dict(im=im, cl=color_bar), code="""
            var o_min = cb_obj.value[0];
            var o_max = cb_obj.value[1];
            im.glyph.color_mapper.low = o_min;
            im.glyph.color_mapper.high = o_max;
            cl.color_mapper.low = o_min;
            cl.color_mapper.high = o_max;
    """)

    ##Change Pallette Selectbox
    palette_select = Select(title="Colormap Select:", options=['Viridis', 'Spectral', 'Inferno'], value='Spectral',
                            callback=callback_color_palette)

    ##Change Color Intensity Slider
    intensity_slider = RangeSlider(title="Color Scale:", start=0, end=2 * np.amax(im1),
                                   value=(0, np.amax(im1)), step=20, )
    intensity_slider.js_on_change('value', callback_color_range)

    ##ROI change
    callback_roi_range = CustomJS(args=dict(mca=mca_source, image=img_source), code="""
            var o_min = cb_obj.value[0];
            var o_max = cb_obj.value[1];
    """)

    roi_slider = RangeSlider(title="Region of Interest:", start=0, end=2560,
                             value=(roi_start * 10, roi_stop * 10), step=10, )
    roi_slider.js_on_change('value', callback_roi_range)

    ##Layout and display
    options = column(det_select, roi_slider, intensity_slider, palette_select, xrf)
    layout = gridplot([[plot, options]])

    show(layout)