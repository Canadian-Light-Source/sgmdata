var x = cb_obj.x;
var y = cb_obj.y;
var co = clip.data['coords'];
var sam = clip.data['sample'];
var edge = clip.data['edges'];
var nscans = clip.data['nscans'];
var type = clip.data['type'];

co.push("(" + x.toString() + ", " + y.toString() + ")");
sam.push("Sample " + sam.length.toString());
edge.push("C");
nscans.push(10);
type.push("sample");
clip.change.emit();

function fallbackCopyTextToClipboard(text) {
  var textArea = document.createElement("textarea");
  textArea.value = text;

  // Avoid scrolling to bottom
  textArea.style.top = "0";
  textArea.style.left = "0";
  textArea.style.position = "fixed";

  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();

  try {
    var successful = document.execCommand('copy');
    var msg = successful ? 'successful' : 'unsuccessful';
    console.log('Fallback: Copying text command was ' + msg);
  } catch (err) {
    console.error('Fallback: Oops, unable to copy', err);
  }

  document.body.removeChild(textArea);
}
function copyTextToClipboard(text) {
  if (!navigator.clipboard) {
    fallbackCopyTextToClipboard(text);
    return;
  }
  navigator.clipboard.writeText(text).then(function() {
    console.log('Async: Copying to clipboard was successful!');
  }, function(err) {
    console.error('Async: Could not copy text: ', err);
  });
}

copyTextToClipboard(co[co.length -1])