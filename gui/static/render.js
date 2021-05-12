window.addEventListener('DOMContentLoaded', function(){
    document.getElementById('src_upload').addEventListener('change', function(e) {
        let filereader = new FileReader();
        filereader.readAsDataURL(e.target.files[0]);
        filereader.addEventListener('load', function(e) {
            console.log(e.target.result)
            var line_element = document.querySelector('.src_result');
            line_element.src = e.target.result;
        });
    });

    document.getElementById('ref_upload').addEventListener('change', function(e) {
        let filereader = new FileReader();
        filereader.readAsDataURL(e.target.files[0]);
        filereader.addEventListener('load', function(e) {
            console.log(e.target.result)
            var line_element = document.querySelector('.ref_result');
            line_element.src = e.target.result;
        });
    });
});

document.querySelector('.manupilate').addEventListener('click', ()=>{
	const src_uri = document.querySelector('.src_result').src;
	const ref_uri = document.querySelector('.ref_result').src;
	const param  = {
		method: 'POST',
		headers: {
		    'Content-Type': 'application/json; charset=utf-8'
		},
		body: JSON.stringify({src_uri: src_uri, ref_uri: ref_uri})
	};
	fetch('/run', param)
		.then((res)=>{
			res.json().then(function(data) {
				var y_element = document.querySelector('.y_result');
				y_element.src = 'data:image/png;base64,' + data['data'];
			});
		});
});