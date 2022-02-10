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