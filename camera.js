/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

 //music: america first: http://gba.wavethemes.net/music-pmd.html
 //quotes: https://www.mcsweeneys.net/articles/the-complete-listing-so-far-atrocities-1-759
 //pose detection: https://github.com/yemount/pose-animator

import * as posenet_module from '@tensorflow-models/posenet';
import * as facemesh_module from '@tensorflow-models/facemesh';
import * as tf from '@tensorflow/tfjs';
import * as paper from 'paper';
import dat from 'dat.gui';
import Stats from 'stats.js';
import "babel-polyfill";

import {drawKeypoints, drawPoint, drawSkeleton, isMobile, toggleLoadingUI, setStatusText} from './utils/demoUtils';
import {SVGUtils} from './utils/svgUtils'
import {PoseIllustration} from './illustrationGen/illustration';
import {Skeleton, facePartName2Index} from './illustrationGen/skeleton';
import {FileUtils} from './utils/fileUtils';

// import * as girlSVG from './resources/illustration/girl.svg';
// import * as boySVG from './resources/illustration/boy.svg';
// import * as abstractSVG from './resources/illustration/abstract.svg';
// import * as blathersSVG from './resources/illustration/blathers.svg';
// import * as tomNookSVG from './resources/illustration/tom-nook.svg';
import * as trumpSVG from './resources/illustration/trump3.svg';

// Camera stream video element
let video;
let videoWidth = 300;
let videoHeight = 300;
let mobileWidth = 800;

// Canvas
let faceDetection = null;
let illustration = null;
let canvasScope;
let canvasWidth = 800;
let canvasHeight = 800;

// ML models
let facemesh;
let posenet;
let minPoseConfidence = 0.15;
let minPartConfidence = 0.1;
let nmsRadius = 30.0;

// Misc
let mobile = false;
const stats = new Stats();
const avatarSvgs = {
  // 'girl': girlSVG.default,
  // 'boy': boySVG.default,
  // 'abstract': abstractSVG.default,
  // 'blathers': blathersSVG.default,
  // 'tom-nook': tomNookSVG.default,
  'trump': trumpSVG.default,
  
};

let $quoteContainer;
let $titleContainer = document.querySelector('.title-container');
let $audio = document.querySelector('.audio');
let $creditsToggle = document.querySelector('.credits-toggle');
let $creditsContainer = document.querySelector('.credits-container');
let $creditsClose = document.querySelector('.credits-close');
let $begin = document.querySelector('.begin');

const allQuotes = [
  '“How amazing, the State Health Director who verified copies of Obama’s “birth certificate”',
  '“When Mexico sends its people,they’re sending people that have lots of problems.”',
  '“I will build a great wall—and nobody builds walls better than me, believe me. I will make Mexico pay for that wall.”',
  '“Happy #CincoDeMayo! The best taco bowls are made in Trump Tower Grill. I love Hispanics!”',
  '“When you’re a star, you can do anything… grab them by the pussy.”',
  '“I need loyalty, I expect loyalty.”',
  '“Go buy Ivanka’s stuff is what I would tell you,”',
  '“I will not be attending the White House Correspondents’ Association Dinner this year. Please wish everyone well and have a great evening!”',
  '“Hey look, in the meantime, I guess, I can’t be doing so badly, because I’m president, and you’re not.”',
  '“I just want to let everybody know in case there was any doubt that we are very much behind President el-Sisi.”',
  '“Someone should look into who paid for the small organized rallies yesterday. The election is over!”',
  '“I would not be as big as I am today without chocolate milk.”',
  '“[I’ve] only been a politician for a short period of time. How am I doing? Am I doing okay? I’m president. Heh! Hey, I’m president!”',
  '“I just fired the head of the FBI. He was crazy, a real nut job…”',
  '“No politician—and I say this with great surety—has been treated worse or more unfairly.”',
  '“Now that Trump won, you’re going to have to go back to Africa, where you belong.”',
  '“enforce the ban on tourism, enforce the embargo,”',
  '“I am being investigated for firing the FBI Director by the man who told me to fire the FBI Director! Witch Hunt”',
  '“Hillary Clinton colluded with the Democratic Party in order to beat Crazy Bernie Sanders.”',
  '“badly bleeding from a face-lift.”',
  '“all have AIDS”',
  '“fucking paranoid schizophrenic;”',
  '“The United States Government will not accept or allow… Transgender individuals to serve in any capacity in the U.S. Military.”',
  '“fire and fury like the world has never seen”',
  '“a hasty withdrawal would create a vacuum for terrorists,”',
  '“I’m a member of your golf club by the way.”',
  '“Fake News.”',
  '“Get that son of a bitch off the field right now, he’s fired. He’s fired!”',
  '“Great solidarity for our National Anthem and for our Country. Standing with locked arms is good, kneeling is not acceptable. Bad ratings!”',
  '“I asked @VP Pence to leave stadium if any players kneeled, disrespecting our country. I am proud of him and @SecondLady Karen.”',
  '“moron,”',
  '“With all of the Fake News coming out of NBC and the Networks, at what point is it appropriate to challenge their License? Bad for country!”',
  '“Don’t ask that guy—he wants to hang them all!”',
  '“Why would Kim Jong-un insult me by calling me ‘old,’ when I would NEVER call him ‘short and fat?’”',
  '“May God be with the people of Sutherland Springs, Texas. The FBI and Law Enforcement has arrived.”',
  '“The Al Frankenstien picture is really bad. Where do his hands go while she sleeps? …..”',
  '“Crooked Hillary Clinton is the worst (and biggest) loser of all time. Hillary, get on with your life and give it another try in three years!”',
  '“Lightweight Senator Kirsten Gillibrand, a total flunky for Chuck Schumer and someone who would come to my office “begging”',
  '“we could use some good old global warming.”',
  '“shithole countries.”',
  '“I’m not afraid of [Muslim people]. I don’t like them. Big difference,”',
  '“Can we call that treason? Why not? I mean, they certainly didn’t seem to love our country very much.”',
  '“[Xi’s] now president for life. I think it’s great. Maybe we’ll have to give that a shot someday.”',
  '“We’re going to be guarding our border with our military. That’s a big step.”',
  '“NEVER, EVER THREATEN THE UNITED STATES AGAIN OR YOU WILL SUFFER CONSEQUENCES THE LIKES OF WHICH FEW THROUGHOUT HISTORY HAVE EVER SUFFERED BEFORE.”',
  '“If anyone is looking for a good lawyer, I would strongly suggest that you don’t retain the services of Michael Cohen!”',
  '“I don’t have an attorney general. It’s very sad,”',
  '“In less than two years, my administration has accomplished more than almost any administration in the history of our country.”',
  '“Look, if we brought George Washington here… the Democrats would vote against him, just so you understand,”',
  '“What neighborhood was it? I don’t know. But I had one beer, that’s the only thing I remember.”',
  '“The paid D.C. protesters are now ready to REALLY protest because they haven’t gotten their checks”',
  '“Any guy that can do a body slam — he’s my kind of guy.”',
  '“Remedy now, or no more Fed payments!”',
  '“First of all, the tear gas is a very minor form of the tear gas itself. It’s very safe.”',
  '“What the hell is going on with Global Waming?”',
  '“hoax”',
  '“There’s nothing wrong with taking information from Russians,”',
  '“Our country is full. We don’t want people coming up here,”',
  '“I’ll say it with great respect, number one, she’s not my type.”',
  '“scumbuckets”',
  '“From day one, my administration has made it a top priority to ensure that America has among the very cleanest air and cleanest water on the planet.”',
  '“We will be ending the AIDS epidemic shortly in America and curing childhood cancer very shortly.”',
  '“Give me your tired and your poor who can stand on their own two feet and who will not become a public charge,”',
  '“Witch Hunt garbage.”',
  '“If Turkey does anything that I, in my great and unmatched wisdom, consider to be off limits, I will totally destroy and obliterate the Economy of Turkey (I’ve done before!).”',
  '“So ridiculous. Greta must work on her Anger Management problem. Chill Greta, Chill!”',
  '“THIS IS AN ASSAULT ON AMERICA, AND AN ASSAULT ON THE REPUBLICAN PARTY!!!!”',
  '“the Obama Admin tried to limit Americans to buying more-expensive LED bulbs for their home.”',
  '“I believed that I needed Yovanovitch out of the way,”',
  '“The Coronavirus is very much under control in the USA,”',
  '“You can’t be a politician and not shake hands,”',
  '“Chinese virus”',
  '“I want them to be appreciative,”',
  '“We’ve now established great testing. … We’ve tested now more than anybody.”',
  '“Time to #FireFauci. You know you’re a fake,”',
  '“LIBERATE MICHIGAN!”',
  '“Because you see it gets in the lungs and it does a tremendous number on the lungs, so it would be interesting to check that,”',
  '“I am not fucking losing to Joe Biden.”',
  '“a lot of the country should be back to normal”',
  '“AMERICA LEADS THE WORLD IN TESTING.”',
  '“Shamu,”',
];

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: videoWidth,
      height: videoHeight,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

const defaultPoseNetArchitecture = 'MobileNetV1';
const defaultQuantBytes = 2;
const defaultMultiplier = 1.0;
const defaultStride = 16;
const defaultInputResolution = 200;

const guiState = {
  avatarSVG: Object.keys(avatarSvgs)[0],
  debug: {
    showDetectionDebug: false,
    showIllustrationDebug: false,
  },
};

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui(cameras) {

  if (cameras.length > 0) {
    guiState.camera = cameras[0].deviceId;
  }

  const gui = new dat.GUI({width: 300});

  let multi = gui.addFolder('Image');
  multi.add(guiState, 'avatarSVG', Object.keys(avatarSvgs)).onChange(() => parseSVG(avatarSvgs[guiState.avatarSVG]));
  multi.open();

  gui.hide(); //MH

  /*
  let output = gui.addFolder('Debug control');
  output.add(guiState.debug, 'showDetectionDebug');
  output.add(guiState.debug, 'showIllustrationDebug');
  output.open();
  */
}

/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
  stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
  document.getElementById('main').appendChild(stats.dom);
}

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video) {
  const canvas = document.getElementById('output');
  const keypointCanvas = document.getElementById('keypoints');
  const videoCtx = canvas.getContext('2d');
  const keypointCtx = keypointCanvas.getContext('2d');

  canvas.width = videoWidth;
  canvas.height = videoHeight;
  keypointCanvas.width = videoWidth;
  keypointCanvas.height = videoHeight;

  async function poseDetectionFrame() {
    // Begin monitoring code for frames per second
    stats.begin();

    let poses = [];
   
    videoCtx.clearRect(0, 0, videoWidth, videoHeight);
    // Draw video
    videoCtx.save();
    videoCtx.scale(-1, 1);
    videoCtx.translate(-videoWidth, 0);
    videoCtx.drawImage(video, 0, 0, videoWidth, videoHeight);
    videoCtx.restore();

    // Creates a tensor from an image
    const input = tf.browser.fromPixels(canvas);
    faceDetection = await facemesh.estimateFaces(input, false, false);
    let all_poses = await posenet.estimatePoses(video, {
      flipHorizontal: true,
      decodingMethod: 'multi-person',
      maxDetections: 1,
      scoreThreshold: minPartConfidence,
      nmsRadius: nmsRadius
    });

    poses = poses.concat(all_poses);
    input.dispose();

    keypointCtx.clearRect(0, 0, videoWidth, videoHeight);
    if (guiState.debug.showDetectionDebug) {
      poses.forEach(({score, keypoints}) => {
      if (score >= minPoseConfidence) {
          drawKeypoints(keypoints, minPartConfidence, keypointCtx);
          drawSkeleton(keypoints, minPartConfidence, keypointCtx);
        }
      });
      faceDetection.forEach(face => {
        Object.values(facePartName2Index).forEach(index => {
            let p = face.scaledMesh[index];
            drawPoint(keypointCtx, p[1], p[0], 2, 'red');
        });
      });
    }

    canvasScope.project.clear();

    if (poses.length >= 1 && illustration) {
      Skeleton.flipPose(poses[0]);

      if (faceDetection && faceDetection.length > 0) {
        let face = Skeleton.toFaceFrame(faceDetection[0]);
        illustration.updateSkeleton(poses[0], face);
      } else {
        illustration.updateSkeleton(poses[0], null);
      }
      illustration.draw(canvasScope, videoWidth, videoHeight);

      if (guiState.debug.showIllustrationDebug) {
        illustration.debugDraw(canvasScope);
      }
    }

    canvasScope.project.activeLayer.scale(
      canvasWidth / videoWidth, 
      canvasHeight / videoHeight, 
      new canvasScope.Point(0, 0));

    // End monitoring code for frames per second
    stats.end();

    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();
}

function setupCanvas() {
  if (window.innerWidth < mobileWidth) {
    canvasWidth = Math.min(window.innerWidth, window.innerHeight);
    canvasHeight = canvasWidth;
    videoWidth *= 0.7;
    videoHeight *= 0.7;
  }  

  canvasScope = paper.default;
  let canvas = document.querySelector('.illustration-canvas');;
  canvas.width = canvasWidth;
  canvas.height = canvasHeight;
  canvasScope.setup(canvas);
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
  setupCanvas();

  toggleLoadingUI(true);
  setStatusText('Loading PoseNet model...');
  posenet = await posenet_module.load({
    architecture: defaultPoseNetArchitecture,
    outputStride: defaultStride,
    inputResolution: defaultInputResolution,
    multiplier: defaultMultiplier,
    quantBytes: defaultQuantBytes
  });
  setStatusText('Loading FaceMesh model...');
  facemesh = await facemesh_module.load();

  setStatusText('Loading Avatar file...');
  let t0 = new Date();
  await parseSVG(Object.values(avatarSvgs)[0]);

  setStatusText('Setting up camera...');
  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this device type is not supported yet, ' +
      'or this browser does not support video capture: ' + e.toString();
    info.style.display = 'block';
    throw e;
  }

  setupGui([], posenet);
  //setupFPS(); //MH
  toggleTitleContainer(true);
  $begin.addEventListener('click',onTitleClick);
  $creditsToggle.addEventListener('click',onCreditsClick);
  $creditsClose.addEventListener('click',onCreditsCloseClick);
  toggleLoadingUI(false);
}
let quoteInterval;

function showQuotes(){
  $quoteContainer = document.querySelector('.quote-container')
  quoteInterval = setInterval(showQuote,5000);
}

function toggleTitleContainer(doShow){
  $titleContainer.classList.toggle('active',doShow);
}

function hideTitleContainer(){
  $titleContainer.classList.toggle('hidden',true);
}

function onCreditsClick(){
  $creditsContainer.classList.toggle('active',true);
}

function onCreditsCloseClick(){
  $creditsContainer.classList.toggle('active',false);
}

function onTitleClick(){
  detectPoseInRealTime(video, posenet);
  showQuotes(); //MH
  toggleAudio(true);
  hideTitleContainer();
}

function toggleAudio(){
  $audio.play();
}

function showQuote(){
  const randomQuote = allQuotes[Math.floor(Math.random() * allQuotes.length)];
  $quoteContainer.innerHTML = randomQuote;
}

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
FileUtils.setDragDropHandler((result) => {parseSVG(result)});

async function parseSVG(target) {
  let svgScope = await SVGUtils.importSVG(target /* SVG string or file path */);
  let skeleton = new Skeleton(svgScope);
  illustration = new PoseIllustration(canvasScope);
  illustration.bindSkeleton(skeleton, svgScope);
}
    
bindPage();
