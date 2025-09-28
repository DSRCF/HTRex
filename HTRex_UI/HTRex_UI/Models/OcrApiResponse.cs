using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace HTRex_UI.Models
{
    public class OcrApiResponse
    {
        [JsonPropertyName("success")]
        public bool Success { get; set; }

        [JsonPropertyName("texts_per_line")]
        public List<string> TextsPerLine { get; set; }

        [JsonPropertyName("original_text")]
        public string OriginalText { get; set; }

        [JsonPropertyName("corrected_text")]
        public string CorrectedText { get; set; }

        [JsonPropertyName("summary_text")]
        public string SummaryText { get; set; }

        [JsonPropertyName("boxes")]
        public List<List<List<int>>> Boxes { get; set; }

        [JsonPropertyName("error")]
        public string Error { get; set; }
    }
}