@echo off
SET RASTER_FILENAME=%1
SET HOST_FILE_PATH=D:\layer-modeller\res\tiff\%RASTER_FILENAME%.tif
SET CONTAINER_PATH=/tmp/%RASTER_FILENAME%
SET CONTAINER_NAME=layer-modeller-server
SET DB_USER=layer-modeller
SET DB_NAME=layer-modeller-database
SET TABLE_NAME=public.%~n1_raster
SET SRID=23700

IF "%RASTER_FILENAME%"=="" (
    echo Error: No raster filename provided.
    echo Usage: import_raster.bat ^<raster_filename^>
    exit /b 1
)
echo Copying %HOST_FILE_PATH% to %CONTAINER_NAME%:%CONTAINER_PATH%...
docker cp "%HOST_FILE_PATH%" "%CONTAINER_NAME%:%CONTAINER_PATH%"

echo Importing %RASTER_FILENAME% into PostgreSQL database...
docker exec -it %CONTAINER_NAME% bash -c "raster2pgsql -s %SRID% %CONTAINER_PATH% -F -t auto %TABLE_NAME% | psql -U %DB_USER% -d %DB_NAME%"
echo Import of %RASTER_FILENAME% completed successfully.