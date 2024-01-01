import React, { useState, useEffect } from "react";
import { Card, Text } from "@mantine/core";
import axios from "axios";
import ImageCard from "./ImageCard";

interface ImageAlbumProps {
  searchDate: string; // In 'YYYYMMDD' format
  flaskEndpoint: string;
}

const ImageAlbum: React.FC<ImageAlbumProps> = ({
  searchDate,
  flaskEndpoint,
}) => {
  const [images, setImages] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    const fetchImages = async () => {
      setLoading(true);
      try {
        const response = await axios.get(
          `${flaskEndpoint}/list_files_by_date`,
          {
            params: { fileName: searchDate },
          }
        );
        setImages(response.data.files);
      } catch (error) {
        console.error("Error fetching images:", error);
      } finally {
        setLoading(false);
      }
    };

    if (searchDate) {
      fetchImages();
    }
  }, [searchDate, flaskEndpoint]);

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <div>
      {images.map((image, index) => (
        <Card key={index}>
          <Text>
            ({index + 1} of {images.length})
          </Text>
          <ImageCard
            flaskEndpoint={flaskEndpoint}
            fileName={image}
            id={index.toString()}
          />
        </Card>
      ))}
    </div>
  );
};

export default ImageAlbum;
