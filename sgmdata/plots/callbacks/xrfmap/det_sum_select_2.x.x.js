const {Float64NDArray} = Bokeh.require("core/util/ndarray");

const fluo_min = sl.value[0] / 10;
const fluo_max = sl.value[1] / 10;
let d = source.data['image'];
const f = det.value;
let cul_image = new Array();

function sumArrays(...arrays) {
  const n = arrays.reduce((max, xs) => Math.max(max, xs.length), 0);
  const result = Float64Array.from({ length: n });
  return result.map((_, i) => arrays.map(xs => xs[i] || 0).reduce((sum, x) => sum + x, 0));
}

if (f.includes('sdd')){
    const image =  sdd.data[f];
    console.log("Summing image for sdd: " + f)
    for (let i = fluo_min; i < fluo_max; i++) {
        cul_image.push(image[i]);
    }
    d[0] = new Float64NDArray(sumArrays(...cul_image), cul_image[0].shape);
}
else if (f === "tey") {
    let tey = source.data['tey'];
    d[0] = tey[0];
}
rect.data['x'][0] = 10*(fluo_min + fluo_max) / 2;
rect.data['width'][0] = Math.abs(fluo_max - fluo_min)*10
rect.change.emit();
source.change.emit();