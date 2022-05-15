//webkitURL is deprecated but nevertheless
URL = window.URL || window.webkitURL;

var gumStream; 						//stream from getUserMedia()
var rec; 							//Recorder.js object
var input; 							//MediaStreamAudioSourceNode we'll be recording

// shim for AudioContext when it's not avb. 
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //audio context to help us record

var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");
var pauseButton = document.getElementById("pauseButton");

//add events to those 2 buttons
recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);
pauseButton.addEventListener("click", pauseRecording);

function startRecording() {
	console.log("recordButton clicked");

	/*
		Simple constraints object, for more advanced audio features see
		https://addpipe.com/blog/audio-constraints-getusermedia/
	*/
    
    var constraints = { audio: true, video:false }

 	/*
    	Disable the record button until we get a success or fail from getUserMedia() 
	*/

	recordButton.disabled = true;
	stopButton.disabled = false;
	pauseButton.disabled = false

	/*
    	We're using the standard promise based getUserMedia() 
    	https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
	*/

	navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
		console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

		/*
			create an audio context after getUserMedia is called
			sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
			the sampleRate defaults to the one set in your OS for your playback device

		*/
		audioContext = new AudioContext();

		//update the format 
		document.getElementById("formats").innerHTML="Format: 1 channel pcm @ "+audioContext.sampleRate/1000+"kHz"

		/*  assign to gumStream for later use  */
		gumStream = stream;
		
		/* use the stream */
		input = audioContext.createMediaStreamSource(stream);

		/* 
			Create the Recorder object and configure to record mono sound (1 channel)
			Recording 2 channels  will double the file size
		*/
		rec = new Recorder(input,{numChannels:1})

		//start the recording process
		rec.record()

		console.log("Recording started");

	}).catch(function(err) {
	  	//enable the record button if getUserMedia() fails
    	recordButton.disabled = false;
    	stopButton.disabled = true;
    	pauseButton.disabled = true
	});
}

function pauseRecording(){
	console.log("pauseButton clicked rec.recording=",rec.recording );
	if (rec.recording){
		//pause
		rec.stop();
		pauseButton.innerHTML="Resume";
	}else{
		//resume
		rec.record()
		pauseButton.innerHTML="Pause";

	}
}

function stopRecording() {
	console.log("stopButton clicked");

	//disable the stop button, enable the record too allow for new recordings
	stopButton.disabled = true;
	recordButton.disabled = false;
	pauseButton.disabled = true;

	//reset button just in case the recording is stopped while paused
	pauseButton.innerHTML="Pause";
	
	//tell the recorder to stop the recording
	rec.stop();

	//stop microphone access
	gumStream.getAudioTracks()[0].stop();

	//create the wav blob and pass it on to createDownloadLink
	rec.exportWAV(createDownloadLink);
}

function createDownloadLink(blob) {

	var url = URL.createObjectURL(blob);
	var au = document.createElement('audio');
	var li = document.createElement('li');
	var link = document.createElement('a');

	//name of .wav file to use during upload and download (without extendion)
	var filename = new Date().toISOString();

	//add controls to the <audio> element
	au.controls = true;
	au.src = url;
	// console.log("au: ", au)
	//save to disk link
	link.href = url;
	link.download = filename+".wav"; //download forces the browser to donwload the file using the  filename
	link.innerHTML = "Save to disk";

	//add the new audio element to li
	li.appendChild(au);

	//add the filename to the li
	li.appendChild(document.createTextNode(filename+".wav "))

	//add the save to disk link to li
	li.appendChild(link);
	li.style = "display: flex; align-items: center;justify-content: center;"

	var upload = document.createElement('button');
	// console.log("Processed")
	upload.className= "btn btn-primary"
	upload.style = "margin-left: 6.3%;padding-right: 3%;padding-left: 3%;"
	upload.href="#";
	upload.innerHTML = "Yêu cầu xử lý";

	token = document.querySelector('meta[name="csrf-token"]').content;
	// console.log("token: ", token)

	// trang thai chon tab
	let tab_status1 = document.getElementById("upload").style.display;
	console.log("tab_status1: ", tab_status1)
	// let model1 = document.getElementById("model1").value;
	let model_record = document.getElementById("model_record").value;
	console.log("model_record: ", model_record)
	upload.addEventListener("click", function(event){
		  var xhr=new XMLHttpRequest();

		  xhr.onload=function(e) {
		      if(this.readyState === 4) {
		          console.log("Server returned: ",e.target.responseText);
		      }
		  };

		  xhr.onreadystatechange = function(){
			if(this.readyState === 4 && this.status === 200)
			  {
				  res = this.response
                const data = JSON.parse(res).data
                const message = JSON.parse(res).message
                console.log("data: ", data)
                // // console.log(message)
                if (message == "Success") {

                    let solved_text = "";
                    for (let i = 0; i < data.length; i++) {
                      solved_text += "&emsp;"+"Từ xuất hiện: " + data[i][0] + "&emsp;&emsp;"+"Độ chính xác: "+ Number((data[i][1]).toFixed(2))+"\n";
                    }
                    document.getElementById("result").innerHTML = solved_text
                } else {
                    // document.getElementById("result").innerHTML = "Vui lòng thử lại"
                    alert("Vui lòng thử lại")
                }
			  }
			};

		  var fd=new FormData();
		  // console.log("blob: ", blob)
		  // console.log("filename: ", filename)
		  fd.append("audio_data",blob, filename);
		  fd.append("tab_status1", tab_status1)
		  fd.append("model", model_record)
		  fd.append('csrfmiddlewaretoken', token);
		  xhr.open("POST","/",true);
		  xhr.send(fd);
	})
	li.appendChild(document.createTextNode (" "))//add a space in between
	li.appendChild(upload)//add the upload link to li

	//add the li element to the ol
	recordingsList.appendChild(li);
}