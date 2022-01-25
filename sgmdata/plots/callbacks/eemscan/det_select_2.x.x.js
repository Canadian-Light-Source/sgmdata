const {Float64NDArray} = Bokeh.require("core/util/ndarray");

const sdd1 = source.data['sdd1'][0];
const sdd2 = source.data['sdd2'][0];
const sdd3 = source.data['sdd3'][0];
const sdd4 = source.data['sdd4'][0];
const xrf_sdd1 = xrf.data['sdd1'];
const xrf_sdd2 = xrf.data['sdd2'];
const xrf_sdd3 = xrf.data['sdd3'];
const xrf_sdd4 = xrf.data['sdd4'];
const d2 = xrf.data;
const d = source.data['image'];
const sum = new Array();
const xrf_sum = new Array();

function sumArrays(...arrays) {
  const n = arrays.reduce((max, xs) => Math.max(max, xs.length), 0);
  const result = Float64Array.from({ length: n });
  return result.map((_, i) => arrays.map(xs => xs[i] || 0).reduce((sum, x) => sum + x, 0));
}

var f = cb_obj.active;
if (f.indexOf(0) > -1) {
    sum.push(sdd1);
    xrf_sum.push(xrf_sdd1);
}
if (f.indexOf(1) > -1) {
    sum.push(sdd2);
    xrf_sum.push(xrf_sdd2);
}
if (f.indexOf(2) > -1) {
    sum.push(sdd3);
    xrf_sum.push(xrf_sdd3);
}
if (f.indexOf(3) > -1) {
    sum.push(sdd4);
    xrf_sum.push(xrf_sdd4);
}
d[0] = new Float64NDArray(sumArrays(...sum), sum[0].shape);
d2['proj_x_tot'] = new Float64NDArray(sumArrays(...xrf_sum), xrf_sum[0].shape);
source.change.emit();
xrf.change.emit();