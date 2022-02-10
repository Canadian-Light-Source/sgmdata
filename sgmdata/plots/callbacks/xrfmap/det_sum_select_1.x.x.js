var fluo_min = sl.value[0] / 10;
var fluo_max = sl.value[1] / 10;
var d = source.data['image'];
var f = det.value;
var cul_image = new Array();

function sumArrays(...arrays) {
  const n = arrays.reduce((max, xs) => Math.max(max, xs.length), 0);
  const result = Float64Array.from({ length: n });
  return result.map((_, i) => arrays.map(xs => xs[i] || 0).reduce((sum, x) => sum + x, 0));
}

if (f.includes('sdd')){
    var image =  sdd.data[f];
    console.log("Summing image for sdd: " + f)
    for (i = fluo_min; i < fluo_max; i++) {
        cul_image.push(image[i]);
    }
    d[0] = sumArrays(...cul_image);

}
if (f === "tey") {
    var tey = source.data['tey'];
    d[0] = tey[0];

}
rect.data['x'][0] = 10*(fluo_min + fluo_max) / 2;
rect.data['width'][0] = Math.abs(fluo_max - fluo_min)*10
rect.change.emit();
source.change.emit();