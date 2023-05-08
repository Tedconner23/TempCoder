class SlidingWindowEncoder:
    def __init__(self, window_size, step_size):
        self.window_size = window_size
        self.step_size = step_size

    def encode(self, text):
        text_windows = []

        # If the text length is smaller than the window_size, return the whole text as a single window
        if len(text) < self.window_size:
            return [text]

        for i in range(0, len(text) - self.window_size, self.step_size):
            text_window = text[i: i + self.window_size]
            text_windows.append(text_window)

        return text_windows

    def decode(self, text_windows):
        # Check if text_windows is empty
        if not text_windows:
            return ""

        # Combine text windows with overlapping regions
        combined_text = text_windows[0]
        for i in range(1, len(text_windows)):
            overlap = self.window_size - self.step_size
            combined_text += text_windows[i][overlap:]

        return combined_text

    def count_characters(self, text):
        return len(text)
