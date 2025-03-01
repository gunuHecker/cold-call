import { useState, useEffect, useRef } from "react";
import axios from "axios";

export default function AudioPage() {
  const [name, setName] = useState("");
  const [purpose, setPurpose] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const nameRef = useRef(""); // Store latest name
  const purposeRef = useRef(""); // Store latest purpose

  // Update refs whenever state changes
  useEffect(() => {
    nameRef.current = name;
    purposeRef.current = purpose;
  }, [name, purpose]);

  useEffect(() => {
    const handleKeyDown = async (event) => {
      if (
        event.code === "Space" &&
        !isRecording &&
        !event.target.matches("input, select")
      ) {
        event.preventDefault(); // Prevents page scrolling
        console.log("Spacebar pressed: Attempting to start recording...");
        await startRecording();
      }
    };

    const handleKeyUp = (event) => {
      if (event.code === "Space" && isRecording) {
        console.log("Spacebar released: Stopping recording...");
        stopRecording();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [isRecording]);

  const startRecording = async () => {
    if (!nameRef.current.trim()) {
      alert("Please enter your name before recording.");
      return;
    }

    if (!purposeRef.current.trim()) {
      alert("Please select a purpose before recording.");
      return;
    }

    try {
      console.log("Requesting microphone access...");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        console.log("Recording stopped. Processing audio...");
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/wav",
        });
        setAudioBlob(audioBlob);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      console.log("Recording started...");
    } catch (error) {
      console.error("Error accessing microphone: ", error);
      alert("Microphone access denied. Please allow microphone permissions.");
    }
  };
  
  const stopRecording = () => {
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      mediaRecorderRef.current.stop();

      // Stop and release the microphone
      mediaRecorderRef.current.stream
        .getTracks()
        .forEach((track) => track.stop());

      setIsRecording(false);
      console.log("Recording stopped and microphone released.");
    }
  };

  const sendAudioToBackend = async () => {
    if (!audioBlob) {
      alert("No audio recorded!");
      return;
    }

    const formData = new FormData();
    formData.append("name", nameRef.current);
    formData.append("purpose", purposeRef.current);
    formData.append("audio", audioBlob, "recording.wav");

    try {
      console.log("Uploading audio...");
      const response = await axios.post("/api/api/v1/processAudio", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      // Axios automatically parses JSON response
      const data = response.data;
      console.log("Response from server: ", data);
    } catch (error) {
      console.error("Error uploading audio: ", error);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-6">
      <h1 className="text-2xl font-bold mb-4">AI Cold Calling Agent</h1>

      <input
        type="text"
        placeholder="Enter your name"
        value={name}
        onChange={(e) => setName(e.target.value)}
        className="p-2 border rounded mb-4 w-full max-w-md"
      />

      <select
        value={purpose}
        onChange={(e) => setPurpose(e.target.value)}
        className="p-2 border rounded mb-4 w-full max-w-md"
      >
        <option value="">Select Purpose</option>
        <option value="demo">Demo Scheduling</option>
        <option value="interview">Candidate Interviewing</option>
        <option value="payment">Payment/Order Follow-up</option>
      </select>

      <p className="text-gray-600 mb-2">
        Press & Hold <b>Spacebar</b> to Record
      </p>
      <button
        className={`px-4 py-2 rounded text-white ${
          isRecording ? "bg-red-500" : "bg-blue-500"
        }`}
        disabled
      >
        {isRecording ? "Recording..." : "Hold Spacebar to Record"}
      </button>

      {audioBlob && (
        <button
          className="mt-4 p-2 bg-green-500 text-white rounded"
          onClick={sendAudioToBackend}
        >
          Send Audio to Backend
        </button>
      )}
    </div>
  );
}
