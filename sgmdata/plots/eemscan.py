from bokeh.layouts import column, row, gridplot
from bokeh.palettes import all_palettes
from bokeh.models import CustomJS, ColumnDataSource, Select, RangeSlider, ColorBar, LinearColorMapper, Rect, Button, \
    CheckboxButtonGroup
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

required = ['image', 'sdd1', 'sdd2', 'sdd3', 'sdd4', 'emission', 'en', 'tey', 'pd', 'io', 'i0']


def plot(**kwargs):
    if 'emission' not in kwargs.keys():
        kwargs['emission'] = np.linspace(0, 2560, 256)
    if 'io' in kwargs.keys() and np.any(kwargs['io']):
        kwargs['i0'] = kwargs['io']
    if "filename" not in kwargs.keys():
        filename = "xas"
    else:
        filename = kwargs['filename']
    source = ColumnDataSource(dict(image=[kwargs['image'].T],
                                   sdd1=[kwargs['sdd1'].T],
                                   sdd2=[kwargs['sdd2'].T],
                                   sdd3=[kwargs['sdd3'].T],
                                   sdd4=[kwargs['sdd4'].T]))

    xrf_source = ColumnDataSource(data=dict(proj_x=np.sum(source.data['image'][0], axis=1),
                                            emission=kwargs['emission'],
                                            proj_x_tot=np.sum(source.data['image'][0], axis=1),
                                            emission_tot=kwargs['emission']))

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
                                            proj_y_tot=np.sum(source.data['image'][0], axis=0)))

    xy_source = ColumnDataSource(data=dict(xaxis=[np.linspace(min(kwargs['en']), max(kwargs['en']), len(kwargs['en']))],
                                           yaxis=[kwargs['emission']]))

    rect_source = ColumnDataSource({'x': [], 'y': [], 'width': [], 'height': []})

    viridis = all_palettes['Viridis'][256]
    inferno = all_palettes['Inferno'][256]
    spectral = all_palettes['Spectral'][11]
    colorblind = all_palettes['Colorblind'][4]

    plot = Figure(plot_width=600, plot_height=600, tools="box_select,save,box_zoom, wheel_zoom,hover,pan,reset")
    color_mapper = LinearColorMapper(palette="Spectral11", low=1, high=np.amax(kwargs['sdd1']))

    delta = max(kwargs['en']) - min(kwargs['en'])
    bins = max(kwargs['emission']) - min(kwargs['emission'])

    im = plot.image(image='image', y=min(kwargs['emission']), x=min(kwargs['en']), dh=bins, dw=delta, source=source,
                    palette="Spectral11")
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, border_line_color=None, location=(0, 0))

    xrf = Figure(plot_width=200, plot_height=600, y_range=plot.y_range, tools="save,hover,box_zoom, pan",
                 title="XRF Projection")
    xrf.circle('proj_x', 'emission', source=xrf_source, alpha=0.6)
    xrf.yaxis.visible = False
    xrf.xaxis.major_label_orientation = "vertical"

    xas = Figure(plot_width=600, plot_height=225, x_range=plot.x_range, tools="save,hover,box_zoom,wheel_zoom,pan",
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

    select_callback = CustomJS(args=dict(s1=source, xrf=xrf_source, xas=xas_source, xy=xy_source, sel=rect_source),
                               code="""
            var rect = sel.data;
            var xarr = xy.data['xaxis'][0];
            var yarr = xy.data['yaxis'][0];
            var d1 = s1.data['image'][0];
            var d2 = xrf.data;
            var d3 = xas.data;
            var xlength = xarr.length;
            var ylength = yarr.length;
            var sum = 0.0;
            if ('geometry' in cb_obj){
                var inds = cb_obj['geometry'];
                if (inds['x1'] > xarr[xarr.length - 1]){
                    inds['x1'] = xarr[xarr.length - 1];
                }
                if (inds['y1'] > yarr[yarr.length - 1]){
                    inds['y1'] = yarr[yarr.length - 1];
                }  
                if (inds['x0'] < xarr[0]){
                    inds['x0'] = xarr[0];
                }
                if (inds['y0'] < yarr[0]){
                    inds['y0'] = yarr[0];
                }    
                rect['x'] = [inds['x0']/2 + inds['x1']/2];
                rect['y'] = [inds['y0']/2 + inds['y1']/2];
                rect['width'] = [inds['x1'] - inds['x0']];
                rect['height'] = [inds['y1'] - inds['y0']];
            }
            else if(rect['x'] && rect['x'].length){
                var inds = {x0: rect['x'][0] - rect['width'][0]/2, x1: rect['x'][0] + rect['width'][0]/2, y0:rect['y'][0] - rect['height'][0]/2, y1:rect['y'][0] + rect['height'][0]/2};

            }
            else if('active' in cb_obj && typeof inds !== 'undefined'){
                inds['y0'] = yarr[0];
                inds['x0'] = xarr[0];
                inds['y1'] = yarr[yarr.length - 1];
                inds['x1'] = xarr[xarr.length - 1];
            }
            else{
                d2['proj_x'] = d2['proj_x_tot'];
                d2['emission'] = d2['emission_tot'];
                d3['en'] = d3['en_tot'];
                d3['proj_y'] = d3['proj_y_tot'];
                xrf.change.emit();
                xas.change.emit();
                sel.change.emit(); 
                return
            }


            function startx(x) {
              return x >= inds['x0'];
            };
            function starty(y){
                return y >= inds['y0'];
            };
            function endx(x){
                return x >= inds['x1'];
            };
            function endy(y){
                return y >= inds['y1'];
            };
            function superslice(arr, start, stop){
                return d1.slice
            }
            d2['proj_x'] = []
            d2['emission'] = []
            d3['proj_y'] = []
            d3['en'] = []
            ystart = yarr.findIndex(starty)
            yend = yarr.findIndex(endy)
            xstart = xarr.findIndex(startx)
            xend = xarr.findIndex(endx)
            d2['emission'] = yarr.slice(ystart,yend);
            for (var i = ystart; i < yend; i++) {
                d2['proj_x'].push(d1.slice(i*xlength+xstart, i*xlength+xend).reduce((a, b) => a + b, 0))
            };
            d3['en'] = xarr.slice(xstart, xend);
            temp = d1.slice(ystart*xlength, yend*xlength);
            for(var i=xstart; i < xend; i++){
                d3['proj_y'].push(
                    temp.filter(function(value, index, Arr){
                        return (index -i) % xlength  == 0;}).reduce((a, b) => a + b, 0));
            };
            xrf.change.emit();
            xas.change.emit();
            sel.change.emit();
        """)

    reset_callback = CustomJS(args=dict(s1=source, xrf=xrf_source, xas=xas_source, xy=xy_source, sel=rect_source), code="""
            var xarr = xy.data['xaxis'][0];
            var yarr = xy.data['yaxis'][0];
            var d1 = s1.data['image'][0];
            var d2 = xrf.data;
            var d3 = xas.data;
            var rect = sel.data;
            rect['x'] = [];
            rect['y'] = [];
            rect['width'] = [];
            rect['height'] = [];
            d2['proj_x'] = d2['proj_x_tot'];
            d2['emission'] = d2['emission_tot'];
            d3['en'] = d3['en_tot'];
            d3['proj_y'] = d3['proj_y_tot'];
            xrf.change.emit();
            xas.change.emit();
            sel.change.emit();
    """)

    plot.js_on_event(events.SelectionGeometry, select_callback)
    plot.js_on_event(events.Reset, reset_callback)

    plot.add_layout(color_bar, 'left')
    callback = CustomJS(args=dict(source=source), code="""
            var sdd1 = source.data['sdd1'][0];
            var sdd2 = source.data['sdd2'][0];
            var sdd3 = source.data['sdd3'][0];
            var sdd4 = source.data['sdd4'][0];
            var d = source.data['image'];
            var sum = new Array();
            function sumArrays(...arrays) {
              const n = arrays.reduce((max, xs) => Math.max(max, xs.length), 0);
              const result = Float64Array.from({ length: n });
              return result.map((_, i) => arrays.map(xs => xs[i] || 0).reduce((sum, x) => sum + x, 0));
            }
            var f = cb_obj.active;
            if (f.indexOf(0) > -1) {
                sum.push(sdd1);
            }
            if (f.indexOf(1) > -1) {
                sum.push(sdd2);
            }
            if (f.indexOf(2) > -1) {
                sum.push(sdd3);
            }
            if (f.indexOf(3) > -1) {
                sum.push(sdd4);
            }
            d[0] = sumArrays(...sum);
            source.change.emit();
    """)
    callback_color_palette = callback_color_range = CustomJS(args=dict(im=im, cl=color_bar), code="""
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
    """ % (viridis, viridis, spectral, spectral, inferno, inferno))

    callback_color_range = CustomJS(args=dict(im=im, cl=color_bar), code="""
            var o_min = cb_obj.value[0];
            var o_max = cb_obj.value[1];
            im.glyph.color_mapper.low = o_min;
            im.glyph.color_mapper.high = o_max;
            cl.color_mapper.low = o_min;
            cl.color_mapper.high = o_max;
    """)

    slider = RangeSlider(title="Color Scale:", start=0, end=10000,
                         value=(0, np.amax(kwargs['sdd1'])), step=20, )
    slider.js_on_change('value', callback_color_range)

    select_palette = Select(title="Colormap Select:", options=['Viridis', 'Spectral', 'Inferno'], value='Spectral')
    select_palette.js_on_change('value', callback_color_palette)

    select = CheckboxButtonGroup(name="Detector Select:", labels=['sdd1', 'sdd2', 'sdd3', 'sdd4'], active=[0])
    select.js_on_change('active', callback, select_callback)

    button = Button(label="Download XAS Spectrum", button_type="success")

    download = CustomJS(args=dict(s2=xas_source, aux=aux_source), code="""
        var sdd = s2.data;
        var aux_data = aux.data;
        var filetext = 'Energy,I0,TEY,Diode,SDD\\n';
        function startx(x){
                return x >= sdd['en'][0];
        };
        xstart = aux_data['en'].findIndex(startx)
        for (i=0; i < sdd['en'].length; i++) {
            if(isNaN(sdd['proj_y'][i])){ continue; }
            else{
                var currRow = [aux_data['en'][i + xstart].toString(),aux_data['i0'][i+xstart].toString(), aux_data['tey'][i + xstart].toString(), aux_data['pd'][i + xstart].toString(), sdd['proj_y'][i].toString().concat('\\n')];
                var joined = currRow.join();
                filetext = filetext.concat(joined);
            }
        }
        var filename = '%s.csv';
        var blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });
        //addresses IE
        if (navigator.msSaveBlob) {
            navigator.msSaveBlob(blob, filename);
        }else {
            var link = document.createElement("a");
            link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            link.target = "_blank";
            link.style.visibility = 'hidden';
            link.dispatchEvent(new MouseEvent('click'));
        }""" % filename)

    button.js_on_event(events.ButtonClick, download)

    options = column(select, button, slider, select_palette)
    layout = gridplot([[xas, options], [plot, xrf]])
    if kwargs.get('json', False):
        return json_item(layout)
    show(layout)