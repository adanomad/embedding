import React from "react";
import { Card, Anchor, Text, Image } from "@mantine/core";

interface SearchResultCardProps {
  fileName: string;
  distance: number;
  flaskEndpoint: string;
  index: number;
}

const SearchResultCard: React.FC<SearchResultCardProps> = ({
  fileName,
  distance,
  flaskEndpoint,
  index,
}) => {
  return (
    <Card key={index}>
      <Anchor href={`${flaskEndpoint}${fileName}`}>
        {index} File Name: {fileName}
      </Anchor>
      <Text>Distance: {distance.toFixed(4)}</Text>
      <Image
        src={`${flaskEndpoint}${fileName}`}
        alt={fileName}
        width={400}
        height={400}
      />
    </Card>
  );
};

export default SearchResultCard;
