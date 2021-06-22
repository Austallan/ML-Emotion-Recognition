using System;
using System.IO;
using System.Collections;
using DlibDotNet;
using DlibDotNet.Extensions;
using Dlib = DlibDotNet.Dlib;

// CMP304: Artificial Intelligence  - Lab 2 Example Code

namespace FeatureExtraction
{
    // The main program class
    class Program
    {
        // file paths
        private const string inputFilePath = @"D:\Abertay Uni Stuff\Computing\3rd year\304\AI part 2\MLAppTestMaker\MLApp2\MLApp2\Cohn-Kanade Images\";

		// The main program entry point
		static void Main(string[] args)
        {
			//The header definition of the csv file
			string header = "label,leftEyebrow,rightEyebrow,leftLip,rightLip,lipHeight,lipWidth,leftEye,rightEye\n";

			//Create the csv file and fill in the first line with the header
			System.IO.File.WriteAllText(@"test_feature_vectors.csv", header);

			string label = "anger";

			for (var n = 0; n < 7; n++)
			{
				if(n == 0)
				{
					label = "anger";
				}
				else if (n == 1)
				{
					label = "disgust";
				}
				else if (n == 2)
				{
					label = "fear";
				}
				else if (n == 3)
				{
					label = "joy";
				}
				else if (n == 4)
				{
					label = "neutral";
				}
				else if (n == 5)
				{
					label = "sadness";
				}
				else if (n == 6)
				{
					label = "surprise";
				}

				string folderSearch = (inputFilePath + label);

				string[] files = Directory.GetFiles(folderSearch);

				for (int f = 0; f < files.Length; f++)
				{

					// Set up Dlib Face Detector
					using (var fd = Dlib.GetFrontalFaceDetector())
					// ... and Dlib Shape Detector
					using (var sp = ShapePredictor.Deserialize(@"D:\Abertay Uni Stuff\Computing\3rd year\304\AI part 2\MLAppTestMaker\MLApp2\MLApp2/shape_predictor_68_face_landmarks.dat"))
					{
						// load input image
						var img = Dlib.LoadImage<RgbPixel>(files[f]);

						Console.WriteLine(label + "\n" + files[f] + "\n");

						//distance calculation
						double distance(FullObjectDetection shape, uint i, uint j)
						{
							return Math.Sqrt(Math.Pow((shape.GetPart(j).X - shape.GetPart(i).X), 2) + Math.Pow((shape.GetPart(j).Y - shape.GetPart(i).Y), 2));
						}


						// find all faces in the image
						var faces = fd.Operator(img);
						// for each face draw over the facial landmarks
						foreach (var face in faces)
						{
							// find the landmark points for this face
							var shape = sp.Detect(img, face);

							// draw the landmark points on the image
							for (var i = 0; i < shape.Parts; i++)
							{
								var point = shape.GetPart((uint)i);
								var rect = new Rectangle(point);

								{
									Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 255, 0), thickness: 4);
								}

							}


							//Left Eyebrow
							var distance1 = distance(shape, 18, 39);
							var distance2 = distance(shape, 19, 39);
							var distance3 = distance(shape, 20, 39);
							var distance4 = distance(shape, 21, 39);

							distance1 = (distance1 / distance4);
							distance2 = (distance2 / distance4);
							distance3 = (distance3 / distance4);
							distance4 = (distance4 / distance4);

							var leftEyebrow = (distance1 + distance2 + distance3 + distance4);

							//Right Eyebrow
							distance1 = distance(shape, 25, 42);
							distance2 = distance(shape, 23, 42);
							distance3 = distance(shape, 24, 42);
							distance4 = distance(shape, 22, 42);

							distance1 = (distance1 / distance4);
							distance2 = (distance2 / distance4);
							distance3 = (distance3 / distance4);
							distance4 = (distance4 / distance4);

							var rightEyebrow = (distance1 + distance2 + distance3 + distance4);


							//Left Lip
							distance1 = distance(shape, 48, 33);
							distance2 = distance(shape, 49, 33);
							distance3 = distance(shape, 50, 33);
							distance4 = distance(shape, 51, 33);

							distance1 = (distance1 / distance4);
							distance2 = (distance2 / distance4);
							distance3 = (distance3 / distance4);
							distance4 = (distance4 / distance4);

							var leftLip = (distance1 + distance2 + distance3);

							//Right Lip
							distance1 = distance(shape, 52, 33);
							distance2 = distance(shape, 53, 33);
							distance3 = distance(shape, 54, 33);
							distance4 = distance(shape, 51, 33);

							distance1 = (distance1 / distance4);
							distance2 = (distance2 / distance4);
							distance3 = (distance3 / distance4);
							distance4 = (distance4 / distance4);

							var rightLip = (distance1 + distance2 + distance3);

							//Lip Width
							var lipWidth = (distance(shape, 48, 54) / distance(shape, 33, 51));

							//Lip Height
							var lipHeight = (distance(shape, 51, 57) / distance(shape, 33, 51));

							//Left Eye
							distance1 = distance(shape, 37, 39);
							distance2 = distance(shape, 38, 39);
							distance3 = distance(shape, 41, 39);
							distance4 = distance(shape, 40, 39);

							distance1 = (distance1 / distance4);
							distance2 = (distance2 / distance4);
							distance3 = (distance3 / distance4);
							distance4 = (distance4 / distance4);

							var leftEye = (distance1 + distance2 + distance3 + distance4);

							//Right Eye
							distance1 = distance(shape, 44, 42);
							distance2 = distance(shape, 43, 42);
							distance3 = distance(shape, 46, 42);
							distance4 = distance(shape, 47, 42);

							distance1 = (distance1 / distance4);
							distance2 = (distance2 / distance4);
							distance3 = (distance3 / distance4);
							distance4 = (distance4 / distance4);

							var rightEye = (distance1 + distance2 + distance3 + distance4);

							//System.Threading.Thread.Sleep(TimeSpan.FromSeconds(10));

							using (System.IO.StreamWriter file = new System.IO.StreamWriter(@"test_feature_vectors.csv", true))
							{
								file.WriteLine(label + "," + leftEyebrow + "," + rightEyebrow + "," + leftLip + "," + rightLip + "," + lipHeight + "," + lipWidth + "," + leftEye + "," + rightEye);
							}
						}

						// export the modified image
						Dlib.SaveJpeg(img, "output.jpg");
					}
				}
			}
		}
    }
}