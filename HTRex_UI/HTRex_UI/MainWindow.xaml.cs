using HTRex_UI.Models;
using MahApps.Metro.Controls;
using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
// using HTRex_UI.Models; // Ensure OcrApiResponse is accessible

namespace HTRex_UI
{
    public partial class MainWindow : MetroWindow
    {
        private string currentImagePath = null;
        private static readonly HttpClient httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };

        private BitmapImage currentBitmap = null;
        private List<List<List<int>>> currentApiBoxes = null;

        private double _currentRotationAngle = 0;
        private double _currentScale = 1.0;
        private const double ZOOM_FACTOR = 1.1;

        private bool _isPanning = false;
        private Point _panLastMousePosition;

        public MainWindow()
        {
            InitializeComponent();
            ResultTextBox.Text = "Please load an image to begin.";
        }

        private void LoadImageButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Title = "Select an Image for HTR",
                Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff|All files (*.*)|*.*"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                try
                {
                    currentImagePath = openFileDialog.FileName;
                    currentBitmap = new BitmapImage();
                    currentBitmap.BeginInit();
                    currentBitmap.UriSource = new Uri(currentImagePath);
                    currentBitmap.CacheOption = BitmapCacheOption.OnLoad;
                    currentBitmap.EndInit();
                    currentBitmap.Freeze();

                    DisplayedImage.Source = currentBitmap;

                    ImageContainerGrid.Width = currentBitmap.PixelWidth;
                    ImageContainerGrid.Height = currentBitmap.PixelHeight;

                    ProcessImageButton.IsEnabled = true;
                    RotateImageButton.IsEnabled = true;
                    ResetViewButton.IsEnabled = true;

                    ResultTextBox.Text = $"Image '{System.IO.Path.GetFileName(currentImagePath)}' loaded. Click 'Recognise Text' to process.";

                    ResetTransformsAndClearBoxes(); // This already clears BoundingBoxCanvas.Children
                    currentApiBoxes = null;
                    ShowBoundingBoxesCheckBox.IsEnabled = false; // Disable if new image is loaded, until new boxes are ready
                    ShowBoundingBoxesCheckBox.IsChecked = false; // Reset to default (hidden)
                }
                catch (Exception ex)
                {
                    MessageBox.Show(this, $"Error loading image: {ex.Message}", "Image Load Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    ProcessImageButton.IsEnabled = false;
                    RotateImageButton.IsEnabled = false;
                    ResetViewButton.IsEnabled = false;
                    currentImagePath = null;
                    currentBitmap = null;
                    DisplayedImage.Source = null;
                    if (ImageContainerGrid != null)
                    {
                        ImageContainerGrid.Width = Double.NaN;
                        ImageContainerGrid.Height = Double.NaN;
                    }
                    ResetTransformsAndClearBoxes();
                    ResultTextBox.Text = "Failed to load image. Please try another file.";
                    ShowBoundingBoxesCheckBox.IsEnabled = false;
                    ShowBoundingBoxesCheckBox.IsChecked = false;
                }
            }
        }

        private async void ProcessImageButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(currentImagePath) || currentBitmap == null)
            {
                MessageBox.Show(this, "Please load an image first.", "No Image Selected", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            SetLoadingState(true);
            BoundingBoxCanvas.Children.Clear();
            ShowBoundingBoxesCheckBox.IsEnabled = false;

            try
            {
                using var form = new MultipartFormDataContent();
                using var fileStream = new FileStream(currentImagePath, FileMode.Open, FileAccess.Read);
                using var streamContent = new StreamContent(fileStream);
                streamContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue($"image/{System.IO.Path.GetExtension(currentImagePath).TrimStart('.')}");
                form.Add(streamContent, "image_file", System.IO.Path.GetFileName(currentImagePath));
                bool useImprovement = UseAiImprovementCheckBox.IsChecked == true;
                form.Add(new StringContent(useImprovement.ToString().ToLower()), "use_improvement");
                ResultTextBox.Text = "Sending image to server and processing...";
                HttpResponseMessage response = await httpClient.PostAsync("http://localhost:5000/recognize", form);

                if (response.IsSuccessStatusCode)
                {
                    string jsonResponse = await response.Content.ReadAsStringAsync();
                    var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                    OcrApiResponse apiResult = JsonSerializer.Deserialize<OcrApiResponse>(jsonResponse, options);

                    if (apiResult != null && apiResult.Success)
                    {
                        StringBuilder sb = new StringBuilder();
                        sb.AppendLine("--- Original Text (Combined) ---");
                        sb.AppendLine(apiResult.OriginalText);
                        if (useImprovement && !string.IsNullOrWhiteSpace(apiResult.CorrectedText))
                        {
                            sb.AppendLine("\n--- AI Corrected Text ---");
                            sb.AppendLine(apiResult.CorrectedText);
                        }
                        if (useImprovement && !string.IsNullOrWhiteSpace(apiResult.SummaryText))
                        {
                            sb.AppendLine("\n--- AI Summary ---");
                            sb.AppendLine(apiResult.SummaryText);
                        }
                        ResultTextBox.Text = sb.ToString();
                        currentApiBoxes = apiResult.Boxes;
                        if (currentApiBoxes != null && currentApiBoxes.Any())
                        {
                            ShowBoundingBoxesCheckBox.IsEnabled = true;
                            DrawBoundingBoxes();
                        }
                        else
                        {
                            ShowBoundingBoxesCheckBox.IsEnabled = false;
                            ShowBoundingBoxesCheckBox.IsChecked = false;
                            BoundingBoxCanvas.Children.Clear();
                        }
                    }
                    else
                    {
                        ResultTextBox.Text = $"API Error: {apiResult?.Error ?? "Unknown API error."}";
                        currentApiBoxes = null;
                    }
                }
                else
                {
                    string errorContent = await response.Content.ReadAsStringAsync();
                    ResultTextBox.Text = $"HTTP Error: {response.StatusCode}\nDetails: {errorContent}";
                    currentApiBoxes = null;
                }
            }
            catch (HttpRequestException httpEx)
            {
                ResultTextBox.Text = $"Network error: {httpEx.Message}. Is the Python AI service running at http://localhost:5000?";
                MessageBox.Show(this, $"Network error: {httpEx.Message}\n\nPlease ensure the Python OCR Service is running.", "Network Error", MessageBoxButton.OK, MessageBoxImage.Error);
                currentApiBoxes = null;
            }
            catch (JsonException jsonEx)
            {
                ResultTextBox.Text = $"JSON Deserialization Error: {jsonEx.Message}. Server response might be invalid.";
                MessageBox.Show(this, $"JSON Deserialization Error: {jsonEx.Message}\n\nServer Response was not valid JSON.", "Response Error", MessageBoxButton.OK, MessageBoxImage.Error);
                currentApiBoxes = null;
            }
            catch (Exception ex)
            {
                ResultTextBox.Text = $"Unexpected error: {ex.Message}";
                MessageBox.Show(this, $"Unexpected error: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                currentApiBoxes = null;
                ShowBoundingBoxesCheckBox.IsEnabled = false;
                ShowBoundingBoxesCheckBox.IsChecked = false;
                BoundingBoxCanvas.Children.Clear();
            }
            finally
            {
                SetLoadingState(false);
                if (currentApiBoxes != null && currentApiBoxes.Any())
                {
                    ShowBoundingBoxesCheckBox.IsEnabled = true;
                }
            }
        }

        private void SetLoadingState(bool isLoading)
        {
           
            LoadingIndicator.IsActive = isLoading;
            ProcessImageButton.IsEnabled = !isLoading && (currentBitmap != null);
            LoadImageButton.IsEnabled = !isLoading;
            UseAiImprovementCheckBox.IsEnabled = !isLoading;
            RotateImageButton.IsEnabled = !isLoading && (currentBitmap != null);
            ResetViewButton.IsEnabled = !isLoading && (currentBitmap != null);
        }

        private void DrawBoundingBoxes()
        {
            BoundingBoxCanvas.Children.Clear();
            if (currentApiBoxes == null || currentBitmap == null)
            {
                return; // Don't draw if no data
            }

            // Only make the canvas visible if the checkbox is checked.
            BoundingBoxCanvas.Visibility = (ShowBoundingBoxesCheckBox.IsChecked == true) ? Visibility.Visible : Visibility.Collapsed;

            foreach (var boxPoints in currentApiBoxes)
            {
                if (boxPoints == null || boxPoints.Count != 4) continue;

                var polygon = new Polygon
                {
                    Stroke = Brushes.Red,
                    StrokeThickness = Math.Max(1, 2 / _currentScale), // Initial thickness
                    Fill = Brushes.Transparent,
                    Points = new PointCollection()
                };

                // Add the four corner points from the API to the polygon
                foreach (var point in boxPoints)
                {
                    if (point.Count == 2)
                    {
                        polygon.Points.Add(new Point(point[0], point[1]));
                    }
                }

                BoundingBoxCanvas.Children.Add(polygon);
            }
        }

        private void ApplyTransforms()
        {
            if (ContentScaleTransform != null && ContentRotateTransform != null)
            {
                ContentScaleTransform.ScaleX = _currentScale;
                ContentScaleTransform.ScaleY = _currentScale;
                ContentRotateTransform.Angle = _currentRotationAngle;
            }
            // Optional: Nudge layout if scrollbars are still misbehaving after other changes
            // ImageScrollViewer?.UpdateLayout(); 

            if (currentApiBoxes != null) DrawBoundingBoxes();
        }

        private void ResetTransformsAndClearBoxes()
        {
            _currentScale = 1.0;
            _currentRotationAngle = 0;

            if (ContentScaleTransform != null)
            {
                ContentScaleTransform.CenterX = 0;
                ContentScaleTransform.CenterY = 0;
            }
            if (ContentRotateTransform != null)
            {
                ContentRotateTransform.CenterX = 0;
                ContentRotateTransform.CenterY = 0;
            }

            ApplyTransforms();
            BoundingBoxCanvas.Children.Clear();
        }

        private void ShowBoundingBoxesCheckBox_Changed(object sender, RoutedEventArgs e)
        {
            // This event fires for both Checked and Unchecked.
            // We just need to re-evaluate whether to draw the boxes.
            if (currentApiBoxes != null && currentBitmap != null) // Only attempt to draw if we have data
            {
                DrawBoundingBoxes();
            }
            else // No data, ensure canvas is clear if checkbox is unchecked
            {
                if (ShowBoundingBoxesCheckBox.IsChecked != true)
                {
                    BoundingBoxCanvas.Children.Clear();
                }
            }
        }

        private void ImageScrollViewer_PreviewMouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (currentBitmap == null || ImageContainerGrid == null || ContentScaleTransform == null) return;
            e.Handled = true;

            Point mousePos = e.GetPosition(ImageContainerGrid);
            double oldScale = _currentScale;

            if (e.Delta > 0) _currentScale *= ZOOM_FACTOR;
            else _currentScale /= ZOOM_FACTOR;
            _currentScale = Math.Max(0.1, Math.Min(_currentScale, 10.0));

            ContentScaleTransform.CenterX = mousePos.X;
            ContentScaleTransform.CenterY = mousePos.Y;

            ApplyTransforms();

            if (ImageScrollViewer != null)
            {
                Point pointOnScaledContentBeforeZoom = new Point(mousePos.X * oldScale, mousePos.Y * oldScale);
                Point pointOnScaledContentAfterZoom = new Point(mousePos.X * _currentScale, mousePos.Y * _currentScale);

                double deltaContentX = pointOnScaledContentAfterZoom.X - pointOnScaledContentBeforeZoom.X;
                double deltaContentY = pointOnScaledContentAfterZoom.Y - pointOnScaledContentBeforeZoom.Y;

                ImageScrollViewer.ScrollToHorizontalOffset(ImageScrollViewer.HorizontalOffset + deltaContentX);
                ImageScrollViewer.ScrollToVerticalOffset(ImageScrollViewer.VerticalOffset + deltaContentY);
            }
        }

        private void RotateImageButton_Click(object sender, RoutedEventArgs e)
        {
            if (currentBitmap == null) return;
            _currentRotationAngle = (_currentRotationAngle - 90) % 360;
            if (ContentRotateTransform != null && ImageContainerGrid != null)
            {
                ContentRotateTransform.CenterX = 0;
                ContentRotateTransform.CenterY = 0;
            }
            ApplyTransforms();
        }

        private void ResetViewButton_Click(object sender, RoutedEventArgs e)
        {
            if (currentBitmap == null) return;
            ResetTransformsAndClearBoxes();
        }

        private void ImageContainerGrid_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (currentBitmap == null || ImageScrollViewer == null) return;
            if (e.ChangedButton == MouseButton.Left)
            {
                _isPanning = true;
                _panLastMousePosition = e.GetPosition(ImageScrollViewer);
                ImageContainerGrid.Cursor = Cursors.ScrollAll;
                ImageContainerGrid.CaptureMouse();
                e.Handled = true;
            }
        }

        private void ImageContainerGrid_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton == MouseButton.Left)
            {
                _isPanning = false;
                ImageContainerGrid.ReleaseMouseCapture();
                ImageContainerGrid.Cursor = Cursors.Arrow;
                e.Handled = true;
            }
        }

        private void ImageContainerGrid_MouseMove(object sender, MouseEventArgs e)
        {
            if (_isPanning && e.LeftButton == MouseButtonState.Pressed && ImageScrollViewer != null)
            {
                Point currentMousePosition = e.GetPosition(ImageScrollViewer);
                double deltaX = currentMousePosition.X - _panLastMousePosition.X;
                double deltaY = currentMousePosition.Y - _panLastMousePosition.Y;
                ImageScrollViewer.ScrollToHorizontalOffset(ImageScrollViewer.HorizontalOffset - deltaX);
                ImageScrollViewer.ScrollToVerticalOffset(ImageScrollViewer.VerticalOffset - deltaY);
                _panLastMousePosition = currentMousePosition;
                e.Handled = true;
            }
        }
    }
}