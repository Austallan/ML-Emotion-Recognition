using System;
using System.IO;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLApp1
{

	public class FaceData
	{
		[LoadColumn(0)]
		public string Label { get; set; }

		[LoadColumn(1)]
		public float LeftEyebrow { get; set; }

		[LoadColumn(2)]
		public float RightEyebrow { get; set; }

		[LoadColumn(3)]
		public float LeftLip { get; set; }

		[LoadColumn(4)]
		public float RightLip { get; set; }

		[LoadColumn(5)]
		public float LipHeight { get; set; }

		[LoadColumn(6)]
		public float LipWidth { get; set; }

		[LoadColumn(7)]
		public float LeftEye { get; set; }

		[LoadColumn(8)]
		public float RightEye { get; set; }
	}

	public class FacePrediction
	{
		[ColumnName("PredictedLabel")]
		public string Label { get; set; }

		[ColumnName("Score")]
		public float[] Score { get; set; }
	}

	class Program
	{
		static void Main(string[] args)
		{
			var mlContext = new MLContext();

			IDataView dataView =
			mlContext.Data.LoadFromTextFile <FaceData>("feature_vectors.csv", hasHeader: true, separatorChar: ',');

			var featureVectorName = "Features";
			var labelColumnName = "Label";
			var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label",
				outputColumnName: labelColumnName)
				.Append(mlContext.Transforms.Concatenate(featureVectorName,
				"LeftEyebrow",
				"RightEyebrow",
				"LeftLip",
				"RightLip",
				"LipHeight",
				"LipWidth",
				"LeftEye",
				"RightEye"))
				.AppendCacheCheckpoint(mlContext)
				.Append(mlContext.MulticlassClassification.Trainers
					.SdcaMaximumEntropy(labelColumnName, featureVectorName))
				.Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
				

			var model = pipeline.Fit(dataView);

			using (var fileStream = new FileStream("faceModel.zip", FileMode.Create, FileAccess.Write, FileShare.Write))

			{ mlContext.Model.Save(model, dataView.Schema, fileStream); }


			/*
			var predictor = mlContext.Model.CreatePredictionEngine<FaceData, FacePrediction> (model);

			var prediction = predictor.Predict(new FaceData()
			{
				LeftEyebrow = 5.48310012876226f,
				RightEyebrow = 5.17659374585107f,
				LeftLip = 4.55653992155817f,
				RightLip = 4.54806557736744f,
				LipHeight = 0.776531624762624f,
				LipWidth = 3.43999428915418f
			});

			Console.WriteLine($"*** Prediction: {prediction.Label} ***");
			Console.WriteLine($"*** Scores: {string.Join(" ", prediction.Score)}");

			*/

			string testDataFileAddress = @"D:\Abertay Uni Stuff\Computing\3rd year\304\AI part 2\MLApp3\test_feature_vectors.csv";

			var testDataView = mlContext.Data.LoadFromTextFile<FaceData>(testDataFileAddress, hasHeader: true, separatorChar: ',');

			var testMetrics = mlContext.MulticlassClassification.Evaluate(model.Transform(testDataView));

			Console.WriteLine($"* Metrics for Multi-class Classification model - Test Data");
			Console.WriteLine($"* MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
			Console.WriteLine($"* MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
			Console.WriteLine($"* LogLoss:          {testMetrics.LogLoss:#.###}");
			Console.WriteLine($"* LogLossReduction: {testMetrics.LogLossReduction:#.###}");
			
			System.Threading.Thread.Sleep(TimeSpan.FromSeconds(30));
		
		}
	}

}

