import React, { useState, useEffect } from "react";
import { Card, Image } from "@mantine/core";
import axios from "axios";

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
          <Image
            src={`${flaskEndpoint}${image}`}
            alt={image}
            width={400}
            height={400}
          />
        </Card>
      ))}
    </div>
  );
};

export default ImageAlbum;
