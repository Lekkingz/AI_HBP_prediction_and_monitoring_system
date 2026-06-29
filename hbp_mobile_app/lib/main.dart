import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {

  runApp(
    const MyApp(),
  );
}

class MyApp extends StatelessWidget {

  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {

    return MaterialApp(

      debugShowCheckedModeBanner: false,

      theme: ThemeData.dark(),

      home: const DashboardPage(),
    );
  }
}

class DashboardPage extends StatefulWidget {

  const DashboardPage({super.key});

  @override
  State<DashboardPage> createState() =>
      _DashboardPageState();
}

class _DashboardPageState
    extends State<DashboardPage> {

  double bp = 0.0;

  int heartRate = 0;

  double temperature = 0.0;

  int respiratoryRate = 0;

  double riskScore = 0.0;

  String riskLevel = "WAITING";


  // ====================================
  // FETCH LIVE DATA
  // ====================================

  Future<void> fetchData() async {

    try {

      final response =
      await http.get(

        Uri.parse(
          'https://ai-hbp-prediction-and-monitoring-system.onrender.com/live-data'
        ),
      );

      if (response.statusCode != 200) {
        print("Server Error: ${response.statusCode}");
        return;
      }

      final data =
      jsonDecode(response.body);

      print(data);

      setState(() {

        bp =
        (data['predicted_systolic_bp']
        as num).toDouble();

        heartRate =
        (data['heart_rate']
        as num).toInt();

        temperature =
        (data['temperature']
        as num).toDouble();

        respiratoryRate =
        (data['respiratory_rate']
        as num).toInt();

        riskScore =
        (data['risk_score']
        as num).toDouble();

        riskLevel =
        data['risk_level']
        .toString();
      });

    } catch (e) {

      print("ERROR:");

      print(e);
    }
  }


  // ====================================
  // INIT
  // ====================================

  @override
  void initState() {

    super.initState();

    fetchData();

    Timer.periodic(

      const Duration(seconds: 2),

      (timer) {

        fetchData();
      },
    );
  }


  // ====================================
  // SENSOR CARD
  // ====================================

  Widget sensorCard(

      IconData icon,
      String title,
      String value
      ) {

    return Container(

      padding:
      const EdgeInsets.all(18),

      decoration: BoxDecoration(

        color:
        const Color(0xff1d2a44),

        borderRadius:
        BorderRadius.circular(20),
      ),

      child: Column(

        crossAxisAlignment:
        CrossAxisAlignment.start,

        children: [

          Icon(

            icon,

            size: 35,

            color: Colors.cyanAccent,
          ),

          const SizedBox(height: 15),

          Text(

            title,

            style: const TextStyle(

              fontSize: 18,

              fontWeight:
              FontWeight.bold,
            ),
          ),

          const SizedBox(height: 10),

          Text(

            value,

            style: const TextStyle(
              fontSize: 22,
            ),
          ),
        ],
      ),
    );
  }


  // ====================================
  // UI
  // ====================================

  @override
  Widget build(BuildContext context) {

    Color riskColor =
    Colors.orange;

    if(riskLevel == "HIGH") {

      riskColor = Colors.red;
    }

    else if(
    riskLevel == "LOW"
    ) {

      riskColor = Colors.green;
    }


    return Scaffold(

      backgroundColor:
      const Color(0xff081229),

      body: SafeArea(

        child: Padding(

          padding:
          const EdgeInsets.all(16),

          child: ListView(

            children: [

              const Text(

                "AI HBP Monitor",

                style: TextStyle(

                  fontSize: 34,

                  fontWeight:
                  FontWeight.bold,
                ),
              ),

              const SizedBox(height: 25),

              Container(

                padding:
                const EdgeInsets.all(24),

                decoration: BoxDecoration(

                  color:
                  const Color(0xff1d2a44),

                  borderRadius:
                  BorderRadius.circular(25),
                ),

                child: Column(

                  crossAxisAlignment:
                  CrossAxisAlignment.start,

                  children: [

                    const Text(

                      "Predicted Systolic BP",

                      style: TextStyle(
                        fontSize: 22,
                      ),
                    ),

                    const SizedBox(height: 20),

                    Text(

                      "${bp.toStringAsFixed(1)} mmHg",

                      style: const TextStyle(

                        fontSize: 60,

                        fontWeight:
                        FontWeight.bold,
                      ),
                    ),

                    const SizedBox(height: 10),

                    Text(

                      riskLevel,

                      style: TextStyle(

                        fontSize: 32,

                        color: riskColor,

                        fontWeight:
                        FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 25),

              GridView.count(

                shrinkWrap: true,

                physics:
                const NeverScrollableScrollPhysics(),

                crossAxisCount: 2,

                crossAxisSpacing: 16,

                mainAxisSpacing: 16,

                childAspectRatio: 1.1,

                children: [

                  sensorCard(

                    Icons.favorite,

                    "Heart Beat Rate",

                    "$heartRate bpm",
                  ),

                  sensorCard(

                    Icons.thermostat,

                    "Temperature",

                    "$temperature °C",
                  ),

                  sensorCard(

                    Icons.air,

                    "Respiratory Rate",

                    "$respiratoryRate breaths/min",
                  ),

                  sensorCard(

                    Icons.analytics,

                    "Risk Score",

                    riskScore.toStringAsFixed(2),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
