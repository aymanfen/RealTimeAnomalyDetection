ThisBuild / scalaVersion := "2.12.18"
ThisBuild / organization := "fraud"
ThisBuild / version      := "1.0.0"

val sparkVersion = "3.5.1"

lazy val root = (project in file("."))
  .settings(
    name := "spark-fraud-stateful",

    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core"            % sparkVersion % "provided",
      "org.apache.spark" %% "spark-sql"              % sparkVersion % "provided",
      "org.apache.spark" %% "spark-streaming"        % sparkVersion % "provided",
      "org.apache.spark" %% "spark-sql-kafka-0-10"   % sparkVersion,
      "org.apache.iceberg" % "iceberg-spark-runtime-3.5_2.12" % "1.4.2",

      // Jackson is already on the Spark executor classpath — mark provided
      // so it is NOT bundled in the fat JAR, avoiding version conflicts.
      // We use it for all JSON serialisation (no circe, no cats, no conflict).
      "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.15.2" % "provided",
      "com.fasterxml.jackson.core"    % "jackson-databind"     % "2.15.2" % "provided",

      "org.slf4j" % "slf4j-api"    % "1.7.36" % "provided",
      "org.slf4j" % "slf4j-simple" % "1.7.36",
    ),

    
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", "MANIFEST.MF")  => MergeStrategy.discard
      case PathList("META-INF", "services", _*) => MergeStrategy.concat
      case PathList("META-INF", _*)             => MergeStrategy.discard
      case PathList("module-info.class")        => MergeStrategy.discard
      case _                                    => MergeStrategy.first
    },
  )
