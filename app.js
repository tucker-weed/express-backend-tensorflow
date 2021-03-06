const express = require('express');
const path = require('path');
const logger = require('morgan');
const cookieParser = require('cookie-parser');
const bodyParser = require('body-parser');
const cors = require('cors');
const {exec} = require('child_process');
const fileUpload = require('express-fileupload');

const index = require('./routes/index');
const users = require('./routes/users');

const app = express();

const imageIds = {}

app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'jade');


app.use(cors());
app.use(logger('dev'));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cookieParser());
app.use(fileUpload());

app.use(express.static(path.join(__dirname, '../client/build')));

app.use('/public', express.static(__dirname + '/public'));

app.use('/', index);

app.post('/upload', (req, res, next) => {
	 let imageFile = req.files.file;
	 var dataToSend;

	// clean zombie resources
	exec('rm *_DONE.jpeg', {
		cwd: './public'
	});

	let key = String(Math.floor((Math.random() * 1000000)));

	while (imageIds[key]) {
	        key = String(Math.floor((Math.random() * 1000000)));
	}
	imageIds[key] = true;
 
	 imageFile.mv(`./public/${key}.jpeg`, err => {
		 if (err) {
			 return res.status(500).send(err);
		 }
	 });
 
 
	 // spawn new child process
	 const python = exec('python3 model.py ' + `${__dirname}/public/${key}.jpeg`, {
		 cwd: './models/xrayModel'
	 });

	 // collect model output
	 python.stdout.on('data', function (data) {
		  dataToSend = data.toString();
	 });

	 // handle response and intermediate stage resource cleansing
	 python.on('close', (code) => {
		 console.log(dataToSend);
		 imageFile.mv(`./public/${key}_DONE.jpeg`, err => {
			if (err) {
				return res.status(500).send(err);
			}
		 });
		exec(`rm ${key}.jpeg`, {
			cwd: './public'
		 });
		 res.json({ name: req.body.filename, file: `public/${key}_DONE.jpeg`, data: dataToSend });
		imageIds[key] = false;
	 });

});

// catch 404
app.use(function(req, res, next) {
	const err = new Error('Not Found');
	err.status = 404;
	next(err);
});

// error handler
app.use(function(err, req, res, next) {
	res.locals.message = err.message;
	console.log(err.message)
	res.locals.error = req.app.get('env') === 'development' ? err : {};
	res.status(err.status || 500).json('Backend');
});

module.exports = app;