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