var rect = sel.data;
var xarr = xy.data['xaxis'][0];
var yarr = xy.data['yaxis'][0];
var d1 = s1.data['image'][0];
var d2 = xrf.data;
var d3 = xas.data;
var xlength = xarr.length;
var sum = 0.0;
var alter = alter.active;
var inds;
var skip = false;

if ('geometry' in cb_obj){
    inds = cb_obj['geometry'];
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
    flslider.value = inds['y0']/2 + inds['y1']/2;
    wdslider.value = inds['y1'] - inds['y0'];
}
else if(rect['x'] && rect['x'].length){
    inds = {x0: rect['x'][0] - rect['width'][0]/2, x1: rect['x'][0] + rect['width'][0]/2, y0:rect['y'][0] - rect['height'][0]/2, y1:rect['y'][0] + rect['height'][0]/2};
}
else{
    det.active = [0];
    alter = 2;
    d2['proj_x'] = d2['proj_x_tot'];
    d2['emission'] = d2['emission_tot'];
    d3['en'] = d3['en_tot'];
    d3['proj_y'] = d3['proj_y_tot'];
    skip = true;
}

if (!skip){
    var i, length;
    let ystart, yend, xstart, xend;
    d3['proj_y'] = []
    d3['en'] = []
    ystart = yarr.findIndex((y) => y >= inds['y0'])
    yend = yarr.findIndex((y) => y >= inds['y1'])
    xstart = xarr.findIndex((x) => x >= inds['x0'])
    xend = xarr.findIndex((x) => x >= inds['x1'])
    d3['en'] = xarr.slice(xstart, xend);
    temp = d1.slice(ystart * xlength, yend * xlength);
    for (i = xstart; i < xend; i++) {
        d3['proj_y'].push(
            temp.filter(function (value, index, Arr) {
                return (index - i) % xlength === 0;
            }).reduce((a, b) => a + b, 0));
    }
    if (alter === 0){
        length = d3['proj_y'].length;
        for(i=1; i < length; i++){
            var last = i - 1;
            var fa = d3['proj_y'][last];
            var fb = d3['proj_y'][i];
            var diff = Math.round(fb-fa);
            let a = d3['en'][last];
            let b = d3['en'][i];
            let add = a + b;
            var diff2 = Math.abs(b - a);
            d3['proj_y'][last] = (diff) / (diff2);
            d3['en'][last] = (add)/ 2;
        }
        d3['proj_y'] = d3['proj_y'].filter((element, index) => {return index < length - 1})
        d3['en'] = d3['en'].filter((element, index) => {return index < length - 1});
    }
    if (alter === 1){
        length = d3['proj_y'].length;
        var y_max = Math.max(...d3['proj_y']);
        var y_min = Math.min(...d3['proj_y']);
        for(i = 0; i < length; i++){
            d3['proj_y'][i] = y_max + (Math.abs(y_max - y_min) / (1 / y_min)) / (d3['proj_y'][i]);
        }
    }
    xrf.change.emit();
    xas.change.emit();
    sel.change.emit();
}


