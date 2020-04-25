/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Demonstrates how to run an audio recognition model in Android.

This example loads a simple speech recognition model trained by the tutorial at
https://www.tensorflow.org/tutorials/audio_training

The model files should be downloaded automatically from the TensorFlow website,
but if you have a custom model you can update the LABEL_FILENAME and
MODEL_FILENAME constants to point to your own files.

The example application displays a list view with all of the known audio labels,
and highlights each one when it thinks it has detected one through the
microphone. The averaging of results to give a more reliable signal happens in
the RecognizeCommands helper class.
*/

package org.tensorflow.lite.examples.speech;

import android.app.Activity;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import androidx.annotation.NonNull;
import androidx.appcompat.widget.SwitchCompat;
import android.util.Log;
import android.view.View;
import android.view.ViewTreeObserver;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import com.google.android.material.bottomsheet.BottomSheetBehavior;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.locks.ReentrantLock;

/*Once you’ve done this you can import a TensorFlow Lite interpreter.
An Interpreter loads a model and allows you to run it, by providing it with a set of inputs.

TensorFlow Lite will then execute the model and write the outputs, it’s really as simple as that.
copied here by Khanh
 */
import org.tensorflow.lite.Interpreter;

/**
 * An activity that listens for audio and then uses a TensorFlow model to detect particular classes,
 * by default a small set of action words.
 */
public class SpeechActivity extends Activity {

  // Constants that control the behavior of the recognition code and model
  // settings. See the audio recognition tutorial for a detailed explanation of
  // all these, but you should customize them to match your training settings if
  // you are running your own model.

  //Model specific constants by Khanh
  private static final int MODEL_SAMPLE_RATE = 16000;
  private static final int MODEL_FRAME_LENGTH = 17920;
  private static final String MODEL_FILEPATH = "file:///android_asset/onsets_frames_wavinput.tflite";
  private static final int INPUT_SAMPLE_RATE = 16000; //TODO We need test this value with real hardware


  // UI elements.
  private static final int REQUEST_RECORD_AUDIO = 13;
  private static final String LOG_TAG = SpeechActivity.class.getSimpleName();
  private static final long MINIMUM_TIME_BETWEEN_SAMPLES_MS = 10;
  private static final short AMPLIFY_PARAM = 1; //TODO should be 1.5 or 2

  // Working variables.
  short[] recordingBuffer = new short[MODEL_FRAME_LENGTH];
  int recordingOffset = 0;
  boolean shouldContinue = true;
  private Thread recordingThread;
  boolean shouldContinueRecognition = true;
  private Thread recognitionThread;
  private final ReentrantLock recordingBufferLock = new ReentrantLock();

  private List<String> labels = new ArrayList<String>();
  private List<String> displayedLabels = new ArrayList<>();

  private long lastProcessingTimeMs;
  private Handler handler = new Handler();
  private TextView selectedTextView = null;
  private HandlerThread backgroundThread;
  private Handler backgroundHandler;

  //Init a new interpreter by Khanh
  private Interpreter tfLite;

  //UI code by Khanh, id = textView
  private TextView outTextView;

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    // Set up the UI.
    super.onCreate(savedInstanceState);
    setContentView(R.layout.tfe_sc_activity_speech);


  //Try to load model by khanh
    String actualModelFilename = MODEL_FILEPATH.split("file:///android_asset/", -1)[1];
    try {
      tfLite = new Interpreter(loadModelFile(getAssets(), actualModelFilename));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    //Setup tensor for input day khanh
    tfLite.resizeInput(0, new int[]{MODEL_FRAME_LENGTH, 1});
    //tfLite.resizeInput(1, new int[]{1});

    // Start the recording and recognition threads.
    requestMicrophonePermission();
    startRecording();
    startRecognition();

    //UI processing code by khanh
    outTextView = findViewById(R.id.textView);
  }

  private void requestMicrophonePermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      requestPermissions(
          new String[] {android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
    }
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    if (requestCode == REQUEST_RECORD_AUDIO
        && grantResults.length > 0
        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
      startRecording();
      startRecognition();
    }
  }

  public synchronized void startRecording() {
    if (recordingThread != null) {
      return;
    }
    shouldContinue = true;
    recordingThread =
        new Thread(
            new Runnable() {
              @Override
              public void run() {
                record();
              }
            });
    recordingThread.start();
  }

  public synchronized void stopRecording() {
    if (recordingThread == null) {
      return;
    }
    shouldContinue = false;
    recordingThread = null;
  }

  private void record() {
    android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

    // Estimate the buffer size we'll need for this device.
    int bufferSize =
        AudioRecord.getMinBufferSize(
                INPUT_SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
    if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
      bufferSize = INPUT_SAMPLE_RATE * 2;
    }
    short[] audioBuffer = new short[bufferSize / 2];

    AudioRecord record =
        new AudioRecord(
            MediaRecorder.AudioSource.DEFAULT,
            INPUT_SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize);

    if (record.getState() != AudioRecord.STATE_INITIALIZED) {
      Log.e(LOG_TAG, "Audio Record can't initialize!");
      return;
    }

    record.startRecording();

    Log.v(LOG_TAG, "Start recording");

    // Loop, gathering audio data and copying it to a round-robin buffer.
    while (shouldContinue) {
      int numberRead = record.read(audioBuffer, 0, audioBuffer.length);
      int maxLength = recordingBuffer.length;
      int newRecordingOffset = recordingOffset + numberRead;
      int secondCopyLength = Math.max(0, newRecordingOffset - maxLength);
      int firstCopyLength = numberRead - secondCopyLength;
      // We store off all the data for the recognition thread to access. The ML
      // thread will copy out of this buffer into its own, while holding the
      // lock, so this should be thread safe.
      recordingBufferLock.lock();
      try {
        System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, firstCopyLength);
        System.arraycopy(audioBuffer, firstCopyLength, recordingBuffer, 0, secondCopyLength);
        recordingOffset = newRecordingOffset % maxLength;
      } finally {
        recordingBufferLock.unlock();
      }
    }

    record.stop();
    record.release();
  }

  public synchronized void startRecognition() {
    if (recognitionThread != null) {
      return;
    }
    shouldContinueRecognition = true;
    recognitionThread =
        new Thread(
            new Runnable() {
              @Override
              public void run() {
                recognize();
              }
            });
    recognitionThread.start();
  }

  public synchronized void stopRecognition() {
    if (recognitionThread == null) {
      return;
    }
    shouldContinueRecognition = false;
    recognitionThread = null;
  }



  private void recognize() {

    Log.v(LOG_TAG, "Start recognition");

    short[] inputBuffer = new short[MODEL_FRAME_LENGTH];
    float[] floatInputBuffer = new float[MODEL_FRAME_LENGTH];

    // Loop, grabbing recorded data and running the recognition model on it.
    while (shouldContinueRecognition) {
      long startTime = new Date().getTime();
      // The recording thread places data in this round-robin buffer, so lock to
      // make sure there's no writing happening and then copy it to our own
      // local version.
      recordingBufferLock.lock();
      try {
        //Phong khi chua recording duoc nhieu (recordingbuffer co it du lieu luc moi bat dau ghi am)
        int maxLength = recordingBuffer.length;
        int firstCopyLength = maxLength - recordingOffset;

        int secondCopyLength = recordingOffset;

        System.arraycopy(recordingBuffer, recordingOffset, inputBuffer, 0, firstCopyLength);
        System.arraycopy(recordingBuffer, 0, inputBuffer, firstCopyLength, secondCopyLength);
      } finally {
        recordingBufferLock.unlock();
      }

      //TODO down sample into model's sample rate, which is 16000
      for (int i = 0; i < MODEL_FRAME_LENGTH; ++i) {
        floatInputBuffer[i] = inputBuffer[i] * AMPLIFY_PARAM;
      }
      float downsample_factor = INPUT_SAMPLE_RATE / MODEL_SAMPLE_RATE;

      //TODO Process mic amplify

      //TODO Extract overlapping



      Object[] inputArray = {floatInputBuffer};
      Map<Integer, Object> output_map = new TreeMap<>();
      float[][][] frame_logits = new float[1][32][88];
      float[][][] onset_logits = new float[1][32][88];
      float[][][] offset_logits = new float[1][32][88];
      float[][][] velocity_values = new float[1][32][88];

      output_map.put(0, frame_logits);
      output_map.put(1, onset_logits);
      output_map.put(2, offset_logits);
      output_map.put(3, velocity_values);

      // Run the model.
      tfLite.runForMultipleInputsOutputs(inputArray, output_map);

      // Use the smoother to figure out if we've had a real recognition event.
      long currentTime = System.currentTimeMillis();
      //final String result = outputScores[0];
      lastProcessingTimeMs = new Date().getTime() - startTime;
      runOnUiThread(
          new Runnable() {
            @Override
            public void run() {
              String outputstring = "";
              StringBuilder sb = new StringBuilder();
              for(int i=0; i<32; i++)
                for(int m=0; m<88; m++)
                {
                  if (onset_logits[0][i][m] > 2)
                  sb.append(String.format("%d-%d: %f \n", (int)(m/12),m%12, onset_logits[0][i][m]));

                }
              outputstring = sb.toString();
              outTextView.setText(String.format("%d ms: %s", lastProcessingTimeMs, outputstring));
            }
          });

      try {
        // We don't need to run too frequently, so snooze for a bit and save some battery.
        Thread.sleep(MINIMUM_TIME_BETWEEN_SAMPLES_MS);
      } catch (InterruptedException e) {
        // Ignore
      }

    }

    Log.v(LOG_TAG, "End recognition");
  }

//  @Override
//  public void onClick(View v) {
//    if (v.getId() == R.id.plus) {
//      String threads = threadsTextView.getText().toString().trim();
//      int numThreads = Integer.parseInt(threads);
//      numThreads++;
//      threadsTextView.setText(String.valueOf(numThreads));
//      //            tfLite.setNumThreads(numThreads);
//      int finalNumThreads = numThreads;
//      backgroundHandler.post(() -> tfLite.setNumThreads(finalNumThreads));
//    } else if (v.getId() == R.id.minus) {
//      String threads = threadsTextView.getText().toString().trim();
//      int numThreads = Integer.parseInt(threads);
//      if (numThreads == 1) {
//        return;
//      }
//      numThreads--;
//      threadsTextView.setText(String.valueOf(numThreads));
//      tfLite.setNumThreads(numThreads);
//      int finalNumThreads = numThreads;
//      backgroundHandler.post(() -> tfLite.setNumThreads(finalNumThreads));
//    }
//  }
//
//  @Override
//  public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
//    backgroundHandler.post(() -> tfLite.setUseNNAPI(isChecked));
//    if (isChecked) apiSwitchCompat.setText("NNAPI");
//    else apiSwitchCompat.setText("TFLITE");
//  }

  private static final String HANDLE_THREAD_NAME = "CameraBackground";

  private void startBackgroundThread() {
    backgroundThread = new HandlerThread(HANDLE_THREAD_NAME);
    backgroundThread.start();
    backgroundHandler = new Handler(backgroundThread.getLooper());
  }

  private void stopBackgroundThread() {
    backgroundThread.quitSafely();
    try {
      backgroundThread.join();
      backgroundThread = null;
      backgroundHandler = null;
    } catch (InterruptedException e) {
      Log.e("amlan", "Interrupted when stopping background thread", e);
    }
  }

  @Override
  protected void onResume() {
    super.onResume();

    startBackgroundThread();
  }

  @Override
  protected void onStop() {
    super.onStop();
    stopBackgroundThread();
  }
}
