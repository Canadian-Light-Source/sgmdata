const x = cb_obj.x;
const y = cb_obj.y;
const co = clip.data['coords'];
const sam = clip.data['sample'];
const edge = clip.data['edges'];
const nscans = clip.data['nscans'];
const type = clip.data['type'];

co.push("(" + x.toString() + ", " + y.toString() + ")");
sam.push("Sample " + sam.length.toString());
edge.push("C");
nscans.push(10);
type.push("sample");
clip.change.emit();

function fallbackCopyTextToClipboard(text) {
  let textArea = document.createElement("textarea");
  textArea.value = text;

  // Avoid scrolling to bottom
  textArea.style.top = "0";
  textArea.style.left = "0";
  textArea.style.position = "fixed";

  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();

  try {
    let successful = document.execCommand('copy');
    let msg = successful ? 'successful' : 'unsuccessful';
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