import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, to_timestamp, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek


config = configparser.ConfigParser()
config.read_file(open('dl.cfg'))

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    '''
    This function creates the Spark session.
    '''
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
        Description: This function reads the songs JSON files from S3 and processes them with Spark.
                     We separate the files into specific dataframes the represent the tables in our star schema model.
                     Then, these tables are saved back to the output folder indicated by output_data parameter.
    
        Parameters:
            spark       : Spark Session
            input_data  : location of song_data json files with the events data
            output_data : S3 bucket were dimensional tables in parquet format will be stored
    """
    # get filepath to song data file
    song_data = os.path.join(input_data,"song_data/*/*/*/*.json")
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df['song_id', 'title', 'artist_id', 'year', 'duration']
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id').parquet(os.path.join(output_data, 'songs.parquet'), 'overwrite')
    print("--- songs.parquet completed ---")

    # extract columns to create artists table
    artists_table = df['artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude']
    
    # write artists table to parquet files
    artists_table.write.parquet(os.path.join(output_data, 'artists.parquet'), 'overwrite')
    print("--- artists.parquet completed ---")
    print("*** process_song_data completed ***\n\n")


def process_log_data(spark, input_data, output_data):
    """
        Description: This function loads log_data from S3 and processes it by extracting the songs and artist tables
                     and then again loaded back to S3. Also output from previous function is used in by spark.read.json command
        
        Parameters:
            spark       : Spark Session
            input_data  : location of log_data json files with the events data
            output_data : S3 bucket were dimensional tables in parquet format will be stored
            
    """
    # get filepath to log data file
    log_data = os.path.join(input_data,"log_data/*.json")

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    actions_df = df.filter(df.page == 'NextSong') \
                   .select('ts', 'userId', 'level', 'song', 'artist',
                           'sessionId', 'location', 'userAgent')
    
    # extract columns for users table    
    users_table = df.select('userId', 'firstName', 'lastName',
                            'gender', 'level').dropDuplicates()
    users_table.createOrReplaceTempView('users')
    # write users table to parquet files
    users_table.write.parquet(os.path.join(output_data, 'users/users.parquet'), 'overwrite')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: str(int(int(x)/1000)))
    actions_df = actions_df.withColumn('timestamp', get_timestamp(actions_df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x) / 1000)))
    actions_df = actions_df.withColumn('datetime', get_datetime(actions_df.ts))
    
    # extract columns to create time table
    time_table = actions_df.select('datetime') \
                           .withColumn('start_time', actions_df.datetime) \
                           .withColumn('hour', hour('datetime')) \
                           .withColumn('day', dayofmonth('datetime')) \
                           .withColumn('week', weekofyear('datetime')) \
                           .withColumn('month', month('datetime')) \
                           .withColumn('year', year('datetime')) \
                           .withColumn('weekday', dayofweek('datetime')) \
                           .dropDuplicates()
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy('year', 'month') \
                    .parquet(os.path.join(output_data,
                                          'time/time.parquet'), 'overwrite')
    
    # read in song data to use for songplays table
    song_data = os.path.join(input_data, "song_data/*/*/*/*.json")
    song_df = spark.read.json(song_data)
    
    # extract columns from joined song and log datasets to create songplays table
    actions_df = actions_df.alias('log_df')
    song_df = song_df.alias('song_df')
    joined_df = actions_df.join(song_df, col('log_df.artist') == col('song_df.artist_name'), 'inner')
    songplays_table = joined_df.select(
        col('log_df.datetime').alias('start_time'),
        col('log_df.userId').alias('user_id'),
        col('log_df.level').alias('level'),
        col('song_df.song_id').alias('song_id'),
        col('song_df.artist_id').alias('artist_id'),
        col('log_df.sessionId').alias('session_id'),
        col('log_df.location').alias('location'), 
        col('log_df.userAgent').alias('user_agent'),
        year('log_df.datetime').alias('year'),
        month('log_df.datetime').alias('month')) \
        .withColumn('songplay_id', monotonically_increasing_id())
    
    songplays_table.createOrReplaceTempView('songplays')
    # write songplays table to parquet files partitioned by year and month
    time_table = time_table.alias('timetable')
    
    songplays_table.write.partitionBy('year', 'month'). \
        parquet(os.path.join(output_data, 'songplays/songplays.parquet'), \
                'overwrite')
    print("--- songplays.parquet completed ---")
    print("*** process_log_data completed ***\n\nEND")



def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    #input_data = "./data/"
    output_data = "./output_data/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
