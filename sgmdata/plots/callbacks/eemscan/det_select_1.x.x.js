var sdd1 = source.data['sdd1'][0];
var sdd2 = source.data['sdd2'][0];
var sdd3 = source.data['sdd3'][0];
var sdd4 = source.data['sdd4'][0];
var xrf_sdd1 = xrf.data['sdd1'];
var xrf_sdd2 = xrf.data['sdd2'];
var xrf_sdd3 = xrf.data['sdd3'];
var xrf_sdd4 = xrf.data['sdd4'];
var d2 = xrf.data;
var d = source.data['image'];
var sum = new Array();
var xrf_sum = new Array();

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
d[0] = sumArrays(...sum);
d2['proj_x_tot'] = sumArrays(...xrf_sum);
source.change.emit();
xrf.change.emit();