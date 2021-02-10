from bokeh.layouts import column, row, gridplot
from bokeh.palettes import all_palettes
from bokeh.models import CustomJS, ColumnDataSource, Select, RangeSlider, ColorBar, LinearColorMapper, Rect, Dropdown
from bokeh.plotting import Figure, output_notebook, output_file, show
from bokeh.embed import json_item
from bokeh import events
from sgmdata.xrffit import gaussians
import numpy as np

required = ['image', 'sdd1', 'sdd2', 'sdd3', 'sdd4', 'tey', 'xp', 'yp', 'emission']


def make_data(df, keys):
    data = {k: df.filter(regex=("%s.*" % k), axis=1).to_numpy() for k in keys}
    data.update({k: np.reshape(v, (len(df.index.levels[0]), len(df.index.levels[1]), v.shape[-1])) if len(
        v.shape) == 2 else np.reshape(v, (len(df.index.levels[0]), len(df.index.levels[1]))) for k, v in data.items()})
    data.update({n: df.index.levels[i] for i, n in enumerate(list(df.index.names))})
    data.update({'emission': sgm_data.scans["2020-01-31t15-01-25-0600"].entry1.fit['emission']})
    data.update({'peaks': sgm_data.scans["2020-01-31t15-01-25-0600"].entry1.fit['peaks']})
    data.update({'width': sgm_data.scans["2020-01-31t15-01-25-0600"].entry1.fit['width']})
    return data


def plot(**kwargs):
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
            var tey = source.data['im5'][0];
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
            if (f == "tey") {
                d[0] = tey;
            }
            source.change.emit();
    """)
    det_select = Select(title="Detector Select:", options=['sdd1', 'sdd2', 'sdd3', 'sdd4', 'tey'], value='sdd1')
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

    if 'peaks' in kwargs.keys() and 'width' in kwargs.keys():
        ##ROI change
        callback_roi_select = CustomJS(args=dict(source=img_source, mca=mca_source, det=det_select), code="""
                var cur = det.properties.value.spec['value'];
                var sel = cb_obj.value[0];
                var data = mca.data;
                var sdd1 = data['sdd1'][sel];
                var sdd2 = data['sdd2'][sel];
                var sdd3 = data['sdd3'][sel];
                var sdd4 = data['sdd4'][sel];
                var img = source.data['image'];
                if (cur == "sdd1") {
                    img[0] = sdd1;
                }
                if (cur == "sdd2") {
                    img[0] = sdd2;
                }
                if (cur == "sdd3") {
                    img[0] = sdd3;
                }
                if (cur == "sdd4") {
                    img[0] = sdd4;
                }
                source.data['im1'][0] = sdd1;
                source.data['im2'][0] = sdd2;
                source.data['im3'][0] = sdd3;
                source.data['im4'][0] = sdd4;
                source.change.emit();
        """)
        roi_menu = [(i, "%.1f" % e) for i, e in enumerate(kwargs['peaks'])]
        roi_slider = Select(title="Fluorescence Line:", options=roi_menu, value="%.1f" % kwargs['peaks'][max_var],
                            callback=callback_roi_select)
        roi_slider.js_on_change('value', callback_roi_select)

        ##Layout and display
        options = column(det_select, intensity_slider, palette_select, xrf, roi_slider)
    else:
        options = column(det_select, intensity_slider, palette_select, xrf)

    layout = gridplot([[plot, options]])
    if kwargs.get('json', False):
        return json_item(layout)
    show(layout)