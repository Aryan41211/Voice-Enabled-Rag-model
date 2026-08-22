import { useCallback, useRef, useState } from "react";

function getRecognitionCtor() {
  if (typeof window === "undefined") {
    return null;
  }

  return (
    window.SpeechRecognition ||
    window.webkitSpeechRecognition ||
    null
  );
}

/*
 * Browser-native speech recognition.
 *
 * Returns the accumulated transcript synchronously from
 * getTranscript(), so callers never race against the
 * recognition onend/onresult events.
 */
function useSpeechRecognition() {
  const [supported] = useState(() => getRecognitionCtor() !== null);
  const [isListening, setIsListening] = useState(false);

  const recognitionRef = useRef(null);
  const transcriptRef = useRef("");
  const langRef = useRef("en-IN");

  const buildTranscript = useCallback((event) => {
    let text = "";

    for (let i = 0; i < event.results.length; i++) {
      text += event.results[i][0].transcript;
    }

    return text.trim();
  }, []);

  const start = useCallback(
    (lang = "en-IN") => {
      const Ctor = getRecognitionCtor();

      if (!Ctor) {
        return false;
      }

      if (recognitionRef.current) {
        try {
          recognitionRef.current.abort();
        } catch (e) {
          // ignore abort errors on restart
        }
        recognitionRef.current = null;
      }

      const recognition = new Ctor();

      recognition.lang = lang;
      langRef.current = lang;
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;

      transcriptRef.current = "";

      recognition.onresult = (event) => {
        transcriptRef.current =
          buildTranscript(event);
      };

      recognition.onerror = () => {
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;

      try {
        recognition.start();
        setIsListening(true);
        return true;
      } catch (e) {
        recognitionRef.current = null;
        setIsListening(false);
        return false;
      }
    },
    [buildTranscript],
  );

  const stop = useCallback(() => {
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch (e) {
        // ignore stop errors
      }
      recognitionRef.current = null;
    }

    setIsListening(false);

    return transcriptRef.current;
  }, []);

  const getTranscript = useCallback(() => {
    return transcriptRef.current;
  }, []);

  return {
    supported,
    isListening,
    start,
    stop,
    getTranscript,
  };
}

export default useSpeechRecognition;
