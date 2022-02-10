const select = sel.data;
const cent = flslider.value;
const wid = wdslider.value;
const rect = fluo.data;
const xarr = xy.data['xaxis'][0];
const yarr = xy.data['yaxis'][0];
const d1 = s1.data['image'][0];
const d2 = xrf.data;
const d3 = xas.data;
const xlength = xarr.length;
const sum = 0.0;
const inds = {x0: xarr[0], x1: xarr[xarr.length -1], y0: cent - wid/2, y1: cent + wid/2};
let max = Math.max(...d2['proj_x']);
let min = Math.min(...d2['proj_x']);
const alt = alter.active;
let ystart, yend, xstart, xend, temp;

rect['y'] = [max/2 + min/2];
rect['x'] = [cent];
rect['height'] = [wid];
rect['width'] =  [max];

if(select['x'] !== undefined && select['x'].length !== 0){
         select['y'] = [cent];
         select['height'] = [wid];
         inds['x0'] = select['x'][0] - select['width'][0]/2;
         inds['x1'] = select['x'][0] + select['width'][0]/2;
}

d3['proj_y'] = []
d3['en'] = []
ystart = yarr.findIndex((y) => y >= inds['y0'])
yend = yarr.findIndex((y) => y >= inds['y1'])
xstart = xarr.findIndex((x) => x >= inds['x0'])
xend = xarr.findIndex((x) => x >= inds['x1'])
d3['en'] = xarr.slice(xstart, xend);
temp = d1.slice(ystart*xlength, yend*xlength);

let i;
for(i=xstart; i < xend; i++){
    d3['proj_y'].push(
        temp.filter(function(value, index, Arr){
            return (index -i) % xlength  === 0;}).reduce((a, b) => a + b, 0));
}
if (alt === 0){
    const length = d3['proj_y'].length;
    for(i = 1; i < length; i++){
        const last = i - 1;
        const fa = d3['proj_y'][last];
        const fb = d3['proj_y'][i];
        const diff = Math.round(fb-fa);
        let a = d3['en'][last];
        let b = d3['en'][i];
        let add = a + b;
        const diff2 = Math.abs(b - a);
        d3['proj_y'][last] = (diff) / (diff2);
        d3['en'][last] = (add)/ 2;
    }
    d3['proj_y'] = d3['proj_y'].filter((element, index) => {return index < length - 1})
    d3['en'] = d3['en'].filter((element, index) => {return index < length - 1});
}
if (alt === 1){
    const length = d3['proj_y'].length;
    const y_max = Math.max(...d3['proj_y']);
    const y_min = Math.min(...d3['proj_y']);
    for(i = 0; i < length; i++){
        d3['proj_y'][i] = y_max + (Math.abs(y_max - y_min) / (1 / y_min)) / (d3['proj_y'][i]);
    }
}
xrf.change.emit();
xas.change.emit();
fluo.change.emit();
sel.change.emit();